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
        self.blocks = [ResBlockEncoder(input_nc if i == 0 else ndf * min(2 ** (i - 1), img_f // ndf),
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

class ResBlockEncoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockEncoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1), use_spect)
        bypass = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(hidden_nc), nonlinearity, conv2,)
        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out