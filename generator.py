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

        gated_convs = self.gated_convs(encoder_1)
        decoder_1 = self.decoder_1(gated_convs)
        output = self.output(decoder_1)
        # x = (x + 1.) / 2. TODO test
        if debug:
            self.debugger['parsing_generator_input'] = x
            self.debugger['parsing_generator_encoder_1'] = encoder_1
            self.debugger['parsing_generator_gated_convs'] = gated_convs
            self.debugger['parsing_generator_decoder_1'] = decoder_1
            self.debugger['parsing_generator_output'] = output
        return output


class Generator(nn.Module):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64,
                     activation=nn.LeakyReLU(), use_spect=True, use_coord=False):
            super(Generator, self).__init__()

            self.debugger = dict()

            self.use_coordconv = True
            self.match_kernel = 3

            self.ngf = ngf


            self.parsing_generator = Parsing_Generator(8 + 18 * 2, 8)

            self.encoder_3 = Encoder_3(3, ngf)  # encoder that gets image S as input
            self.per_region_encoding = Per_Region_Encoding(ngf)


            self.vgg19 = Vgg_Encoder()
            # self.getMatrix = GetMatrix(ngf * 4, 1)

            self.phi = nn.Conv2d(in_channels=ngf * 4 + 3, out_channels=ngf * 4, kernel_size=1, stride=1, padding=0)
            self.theta = nn.Conv2d(in_channels=ngf * 4 + 3, out_channels=ngf * 4, kernel_size=1, stride=1, padding=0)

            self.encoder_2 = Encoder_2(8 + 18 + 8 + 3, ngf)  # encoder that gets parsing and 3 other inputs

            self.decoder_2 = Decoder_2(3, ngf)

            self.per_region_normalization = Per_Region_Normalization(ngf * 4, 256)  # spatial aware normalization $ per region normalization

            self.gamma = nn.Conv2d(self.ngf * 4, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.beta = nn.Conv2d(self.ngf * 4, 1, kernel_size=1, stride=1, padding=0, bias=False)

            self.res_block = Res_Block(ngf * 4, output_nc=ngf * 4, hidden_nc=ngf * 4, norm_layer=nn.InstanceNorm2d)

            self.res_block2 = Res_Block(ngf * 4, output_nc=ngf * 4, hidden_nc=ngf * 4, norm_layer=nn.InstanceNorm2d)
            # self.res = ResBlock(ngf * 4, output_nc=ngf * 4, hidden_nc=ngf * 4, norm_layer=norm_layer,
            #                     nonlinearity=activation,
            #                     learnable_shortcut=False, use_spect=False, use_coord=False)
            #
            # self.res1 = ResBlock(ngf * 4, output_nc=ngf * 4, hidden_nc=ngf * 4, norm_layer=norm_layer,
            #                      nonlinearity=activation,
            #                      learnable_shortcut=False, use_spect=False, use_coord=False)


    def forward(self, img1, img2, pose1, pose2, par1, par2, debug=False, use_coord=True): # TODO overall
        encoder_3 = self.encoder_3(img1, debug=debug)
        codes_vector, exist_vector, img1code = self.per_region_encoding(encoder_3, par1, debug=debug)  # Fi

        parcode = self.parsing_generator(torch.cat((par1, pose1, pose2), 1), debug=debug)  # parsing output
        par2 = parcode

        parcode = self.encoder_2(torch.cat((par1, par2, pose2, img1), 1), debug=debug)  # Fp


        parcode = self.per_region_normalization(parcode, par2, codes_vector, exist_vector, debug=debug)  # Fp

        res = self.res_block(parcode, debug=debug)

        parcode = res
        ## regularization to let transformed code and target image code in the same feature space

        img2code = self.vgg19(img2)  # VGG output of original image
        loss_reg = F.mse_loss(img2code, parcode)



        # -------------

        img1_vgg = self.vgg19(img1)

        parcode1_norm = torch.div(parcode, torch.norm(parcode, 2, 1, keepdim=True) + sys.float_info.epsilon)
        img1code_norm = torch.div(img1_vgg, torch.norm(img1_vgg, 2, 1, keepdim=True) + sys.float_info.epsilon)

        if use_coord:
            parcode1 = add_coords(parcode1_norm)
            img1code1 = add_coords(img1code_norm)

        gamma = self.gamma(img1code_norm)
        beta = self.beta(img1code_norm)

        batch_size, _, h, w = gamma.shape
        print(img1code.shape, parcode.shape, img1code1.shape, parcode1.shape)
        # print(img1code1.shape, parcode1.shape)
        att = self.computecorrespondence(parcode1, img1code1)

        imgamma = torch.bmm(gamma.view(batch_size, 1, -1), att).view(batch_size, 1, h, w).contiguous()
        imbeta = torch.bmm(beta.view(batch_size, 1, -1), att).view(batch_size, 1, h, w).contiguous()


        parcode = parcode * (1 + imgamma) + imbeta
        res2 = self.res_block2(parcode, debug=debug)


        # -------------

        parcode = self.decoder_2(res2, debug=debug)
        if debug:
            self.debugger['generator_img1'] = img1
            self.debugger['generator_img2'] = img2
            self.debugger['generator_pose1'] = pose1
            self.debugger['generator_pose2'] = pose2
            self.debugger['generator_par1'] = par1
            self.debugger['generator_par2'] = par2
            self.debugger['generator_encoder_3'] = encoder_3
            self.debugger['generator_codes_vector'] = codes_vector
            self.debugger['generator_exist_vector'] = exist_vector
            self.debugger['generator_img1code'] = img1code
            self.debugger['generator_output'] = parcode
            self.debugger['generator_img2code'] = img2code
            self.debugger['generator_loss_reg'] = loss_reg
            self.debugger['generator_parcode1_norm'] = parcode1_norm
            self.debugger['generator_img1code_norm'] = img1code_norm
            self.debugger['generator_gamma'] = gamma
            self.debugger['generator_beta'] = beta
            self.debugger['generator_att'] = att
            self.debugger['generator_imgamma'] = imgamma
            self.debugger['generator_imbeta'] = imbeta
            self.debugger['generator_parcode1'] = parcode1
            self.debugger['generator_img1code1'] = img1code1
            self.debugger['res2'] = res2
            self.debugger['res'] = res




        return parcode, loss_reg, par2

    def computecorrespondence(self, fea1, fea2, temperature=0.01, detach_flag=False, WTA_scale_weight=1, alpha=1):
        batch_size, channel_size = fea2.shape[:2]

        # Helper function to process and normalize features
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
        theta = _process_feature(fea1, self.match_kernel, self.theta) #maybe it bshould be fea2
        phi = _process_feature(fea2, self.match_kernel, self.phi)

        # Compute correspondence and apply weight-temperature scaling if needed
        f = torch.matmul(theta.permute(0, 2, 1), phi)
        if WTA_scale_weight != 1:
            f = WTA_scale.apply(f, WTA_scale_weight)
        f /= temperature

        # Apply softmax to get attention
        att = F.softmax(f.permute(0, 2, 1), dim=-1)
        return att

    def print_network(self):
        """Print out the network information"""
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('The number of parameters: {}'.format(num_params))





def add_coords(x):
    bs, _, h, w = x.shape
    xx_range = torch.linspace(-1, 1, steps=w, dtype=x.dtype, device=x.device).repeat(bs, h, 1).unsqueeze(1)
    yy_range = torch.linspace(-1, 1, steps=h, dtype=x.dtype, device=x.device).repeat(bs, w, 1).transpose(1, 2).unsqueeze(1)
    rr_channel = torch.sqrt(xx_range.pow(2) + yy_range.pow(2))

    return torch.cat((x, xx_range, yy_range, rr_channel), dim=1)



