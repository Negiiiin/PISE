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

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm_layer=None, activation=nn.LeakyReLU(0.1),
                 use_spect=True):
        super(Discriminator, self).__init__()
        self.layers = layers
        self.activation = activation
        self.debugger = dict()
        self.blocks = [Res_Block_Encoder(input_nc if i == 0 else ndf * min(2 ** (i - 1), img_f // ndf),
                                       ndf * min(2 ** i, img_f // ndf),
                                       ndf * min(2 ** (i - 1), img_f // ndf) if i > 0 else ndf,
                                       norm_layer, activation, use_spect) for i in range(layers)]

        self.blocks = nn.ModuleList(self.blocks)
        self.final_conv = spectral_norm(nn.Conv2d(ndf * min(2 ** (layers - 1), img_f // ndf), 1, 1))

    def forward(self, x, debug=False):
        if debug:
            self.debugger[f'Input'] = x
        for index, block in enumerate(self.blocks):
            x = block(x)
            if debug:
                self.debugger[f'layer{index}'] = x
        out = self.final_conv(self.activation(x))
        if debug:
            self.debugger[f'Output'] = out
        return out