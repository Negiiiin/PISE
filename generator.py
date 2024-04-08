import functools

import torch.nn as nn
import torch
from torch.nn.utils.spectral_norm import spectral_norm
import torchvision
import os
import torch.nn.functional as F
import loss
import numpy as np
from PIL import Image
import ntpath
from torch.nn import init
import warnings
from torch.optim import lr_scheduler

from VGG19 import VGG19

from basic_blocks import *
import sys

class Parsing_Generator(nn.Module):
    """
        Hard Encoder with configurable normalization, activation, and spectral normalization.
        Uses EncoderBlocks and ResBlockDecoders for encoding, and Gated Convolutions for feature modulation.
    """
    def __init__(self, input_nc, generator_filter_num=32, norm_layer=nn.InstanceNorm2d,
                 activation=nn.LeakyReLU(0.2), use_spect=False, kernel_size=3):
        super(Parsing_Generator, self).__init__()

        self.debugger = dict()
        # Define encoder blocks
        self.encoder_1 = Encoder_1(input_nc, generator_filter_num, norm_layer, activation, use_spect)


        # Define gated convolutions
        self.gated_convs = nn.Sequential(
            Gated_Conv(generator_filter_num*16, generator_filter_num*16),
            Gated_Conv(generator_filter_num*16, generator_filter_num*16)
        )

        # Define residual blocks in the decoder
        self.decoder_1 = Decoder_1(input_nc, generator_filter_num, norm_layer, activation, use_spect)

        kwargs = {'kernel_size': kernel_size, 'padding': 0, 'bias': True}


        self.output = nn.Sequential(
            norm_layer(generator_filter_num),
            activation,
            nn.ReflectionPad2d(int((kernel_size - 1) / 2)),
            Coord_Conv(generator_filter_num, 8, use_spect=use_spect, **kwargs),
            nn.Tanh()
        )

    def forward(self, x, debug=False):
        encoder_1 = self.encoder_1(x)

        for gate in self.gated_convs:
            encoder_1 = gate(encoder_1, self.debugger)
        # gated_convs = self.gated_convs(encoder_1)
        gated_convs = encoder_1
        decoder_1 = self.decoder_1(gated_convs)
        output = self.output(decoder_1)
        output = (output + 1.) / 2. # TODO test
        if debug:
            self.debugger['parsing_generator_input'] = x
            self.debugger['parsing_generator_encoder_1'] = encoder_1
            self.debugger['parsing_generator_gated_convs'] = gated_convs
            self.debugger['parsing_generator_decoder_1'] = decoder_1
            self.debugger['parsing_generator_output'] = output

        return output


class Generator(nn.Module):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, activation=nn.LeakyReLU(), use_spect=True):
        super(Generator, self).__init__()

        # Configuration parameters stored as attributes
        self.ngf = ngf
        self.image_nc = image_nc
        self.structure_nc = structure_nc
        self.output_nc = output_nc
        self.activation = activation
        self.use_spect = use_spect
        self.debugger = dict()
        self.match_kernel = 3
        # Define network components
        self.parsing_generator = Parsing_Generator(8 + self.structure_nc * 2, 8)
        self.encoder_3 = Encoder_3(self.image_nc, self.ngf)
        self.per_region_encoding = Per_Region_Encoding(self.ngf)
        self.vgg19 = Vgg_Encoder()
        self.encoder_2 = Encoder_2(8 + self.structure_nc + 8 + self.image_nc, self.ngf)
        self.decoder_2 = Decoder_2(self.image_nc, self.ngf)
        self.per_region_normalization = Per_Region_Normalization(self.ngf * 4, 256)

        # Convolutions for feature transformations
        self.phi = nn.Conv2d(in_channels=self.ngf * 4 + self.image_nc, out_channels=self.ngf * 4, kernel_size=1)
        self.theta = nn.Conv2d(in_channels=self.ngf * 4 + self.image_nc, out_channels=self.ngf * 4, kernel_size=1)
        self.gamma = nn.Conv2d(self.ngf * 4, 1, kernel_size=1, bias=False)
        self.beta = nn.Conv2d(self.ngf * 4, 1, kernel_size=1, bias=False)

        # Residual blocks
        self.res_block = Res_Block(self.ngf * 4, output_nc=self.ngf * 4, hidden_nc=self.ngf * 4,
                                   norm_layer=nn.InstanceNorm2d)
        self.res_block2 = Res_Block(self.ngf * 4, output_nc=self.ngf * 4, hidden_nc=self.ngf * 4,
                                    norm_layer=nn.InstanceNorm2d)

    def forward(self, input_image_s, target_image, pose1, pose2, input_segmentation_s, target_segmentation=None, debug=False, use_coord=True): # input_image_s, target_image, pose1, pose2, input_segmentation_s, parsing_segmentation_g
        # Encode the first image
        encoder_3_output = self.encoder_3(input_image_s, debug=debug)

        # Generate per-region encoding
        codes_vector, exist_vector = self.per_region_encoding(encoder_3_output, input_segmentation_s, debug=debug)

        # Generate parsing code
        combined_input = torch.cat((input_segmentation_s, pose1, pose2), 1)
        parsing_segmentation_g = self.parsing_generator(combined_input, debug=debug)

        # Further encode the parsing code along with other inputs
        encoder_2_input = torch.cat((input_segmentation_s, parsing_segmentation_g, pose2, input_image_s), 1)
        encoder_2_output = self.encoder_2(encoder_2_input, debug=debug)

        # Apply per-region normalization to the encoded parsing code
        per_region_normalization_out = self.per_region_normalization(encoder_2_output, parsing_segmentation_g, codes_vector, exist_vector, debug=debug)

        # Process the normalized parsing code through the first residual block
        res_block_1_output = self.res_block(per_region_normalization_out, debug=debug)

        # Compare the transformation code with the VGG features of the second image
        img2code = self.vgg19(target_image)  # VGG features of target_image
        loss_reg = F.mse_loss(img2code, res_block_1_output)

        # Normalize parcode and VGG features of input_image_s for correspondence computation
        res_block_1_output_norm = res_block_1_output / (torch.norm(res_block_1_output, p=2, dim=1, keepdim=True) + sys.float_info.epsilon)
        img1_vgg = self.vgg19(input_image_s)
        img1_vgg_norm = img1_vgg / (torch.norm(img1_vgg, p=2, dim=1, keepdim=True) + sys.float_info.epsilon)

        # Compute spatial attention
        gamma = self.gamma(img1_vgg_norm)
        beta = self.beta(img1_vgg_norm)

        # Optionally add coordinate channels
        if use_coord:
            res_block_1_output_norm = add_coords(res_block_1_output_norm)
            img1_vgg_norm = add_coords(img1_vgg_norm)


        att = self.compute_correspondence(res_block_1_output_norm, img1_vgg_norm)
        batch_size, _, h, w = gamma.shape
        imgamma = torch.bmm(gamma.view(batch_size, 1, -1), att).view(batch_size, 1, h, w).contiguous()
        imbeta = torch.bmm(beta.view(batch_size, 1, -1), att).view(batch_size, 1, h, w).contiguous()

        # Apply spatial attention to the parsing code
        res_block_1_output = res_block_1_output * (1 + imgamma) + imbeta

        # Process the adjusted parsing code through the second residual block
        res_block_2_output = self.res_block2(res_block_1_output, debug=debug)
        generated_image = self.decoder_2(res_block_2_output, debug=debug)

        # Return the final transformed code
        return generated_image, loss_reg, parsing_segmentation_g


    def compute_correspondence(self, fea1, fea2, temperature=0.01):
            batch_size, channel_size = fea2.shape[:2]

            def _process_feature(feature, kernel, phi_or_theta):
                transformed = phi_or_theta(feature)
                if kernel == 1:
                    transformed = transformed.view(batch_size, channel_size, -1)
                else:
                    transformed = F.unfold(transformed, kernel_size=kernel, padding=kernel // 2)

                # Normalize feature
                transformed -= transformed.mean(dim=1, keepdim=True)
                norm = torch.norm(transformed, p=2, dim=1, keepdim=True) + torch.finfo(transformed.dtype).eps
                return transformed / norm

            # Process features
            theta = _process_feature(fea1, self.match_kernel, self.theta)
            phi = _process_feature(fea2, self.match_kernel, self.phi)

            f = torch.matmul(theta.permute(0, 2, 1), phi)
            f /= temperature

            att = F.softmax(f.permute(0, 2, 1), dim=-1)
            return att


def add_coords(x):
    bs, _, h, w = x.shape
    xx_range = torch.linspace(-1, 1, steps=w, dtype=x.dtype, device=x.device).repeat(bs, h, 1).unsqueeze(1)
    yy_range = torch.linspace(-1, 1, steps=h, dtype=x.dtype, device=x.device).repeat(bs, w, 1).transpose(1, 2).unsqueeze(1)
    rr_channel = torch.sqrt(xx_range.pow(2) + yy_range.pow(2))

    return torch.cat((x, xx_range, yy_range, rr_channel), dim=1)