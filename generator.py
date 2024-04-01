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

class Parsing_Generator(nn.Module):
    """
        Hard Encoder with configurable normalization, activation, and spectral normalization.
        Uses EncoderBlocks and ResBlockDecoders for encoding, and Gated Convolutions for feature modulation.
    """
    def __init__(self, input_nc, generator_filter_num=32, norm_layer=nn.BatchNorm2d,
                 activation=nn.LeakyReLU(0.1), use_spect=True, kernel_size=3):
        super(Parsing_Generator, self).__init__()

        # Define encoder blocks
        self.encoder_1 = Encoder_1(input_nc, generator_filter_num, norm_layer, activation, use_spect)


        # Define residual blocks in the decoder
        self.decoder_1 = Decoder_1(input_nc, generator_filter_num, norm_layer, activation, use_spect)


        # Define gated convolutions
        self.gated_convs = nn.Sequential(
            Gated_Conv(generator_filter_num*16, generator_filter_num*16),
            Gated_Conv(generator_filter_num*16, generator_filter_num*16)
        )

        kwargs = {'kernel_size': kernel_size, 'padding': 0, 'bias': True}


        self.output = nn.Sequential(
            norm_layer(generator_filter_num),
            activation,
            nn.ReflectionPad2d(int((kernel_size - 1) / 2)),
            Coord_Conv(generator_filter_num, 8, use_spect=use_spect, **kwargs),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder_1(x)
        x = self.gated_convs(x)
        x = self.decoder_1(x)
        x = self.output(x)
        # x = (x + 1.) / 2. TODO test
        return x


class Generator(nn.Module):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64,
                     activation=nn.LeakyReLU(), use_spect=True, use_coord=False):
            super(Generator, self).__init__()

            self.use_coordconv = True
            self.match_kernel = 3


            self.parsing_generator = Parsing_Generator(8 + 18 * 2, 8)

            self.encoder_3 = Encoder_3(3, ngf)  # encoder that gets image S as input
            self.per_region_encoding = Per_Region_Encoding(ngf)


            self.vgg19 = Vgg_Encoder()
            # self.getMatrix = GetMatrix(ngf * 4, 1)

            # self.phi = nn.Conv2d(in_channels=ngf * 4 + 3, out_channels=ngf * 4, kernel_size=1, stride=1, padding=0)
            # self.theta = nn.Conv2d(in_channels=ngf * 4 + 3, out_channels=ngf * 4, kernel_size=1, stride=1, padding=0)

            self.encoder_2 = Encoder_2(8 + 18 + 8 + 3, ngf)  # encoder that gets parsing and 3 other inputs

            self.decoder_2 = Decoder_2(3, ngf)

            self.per_region_normalization = Per_Region_Normalization(ngf * 4, 256)  # spatial aware normalization $ per region normalization
            # self.res = ResBlock(ngf * 4, output_nc=ngf * 4, hidden_nc=ngf * 4, norm_layer=norm_layer,
            #                     nonlinearity=activation,
            #                     learnable_shortcut=False, use_spect=False, use_coord=False)
            #
            # self.res1 = ResBlock(ngf * 4, output_nc=ngf * 4, hidden_nc=ngf * 4, norm_layer=norm_layer,
            #                      nonlinearity=activation,
            #                      learnable_shortcut=False, use_spect=False, use_coord=False)


    def forward(self, img1, img2, pose1, pose2, par1, par2):
        encoder_3 = self.encoder_3(img1)
        codes_vector, exist_vector, img1code = self.per_region_encoding(encoder_3, par1)  # Fi

        parcode = self.parsing_generator(torch.cat((par1, pose1, pose2), 1))  # parsing output
        par2 = parcode

        parcode = self.encoder_2(torch.cat((par1, par2, pose2, img1), 1))  # Fp


        parcode = self.per_region_normalization(parcode, par2, codes_vector, exist_vector)
        # parcode = self.res(parcode)

        ## regularization to let transformed code and target image code in the same feature space

        img2code = self.vgg19(img2)  # VGG output of original image
        loss_reg = F.mse_loss(img2code, parcode)

        parcode = self.decoder_2(parcode)
        return parcode, loss_reg, par2