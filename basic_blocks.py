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

class Coord_Conv(nn.Module):
    """
        This class adds coordinate channels to the input tensor.
    """
    def __init__(self, conv_in_channel, conv_out_channel, use_spect=False, has_radial_dist=False, **kwargs):
        super(Coord_Conv, self).__init__()
        self.has_radial_dist = has_radial_dist
        self.conv_in_channel = conv_in_channel + 2 + (1 if has_radial_dist else 0)
        self.conv = spectral_norm(nn.Conv2d(self.conv_in_channel, conv_out_channel, **kwargs)) if use_spect \
            else nn.Conv2d(self.conv_in_channel, conv_out_channel, **kwargs)

    def forward(self, x):
        """
            input:  Input tensor with shape (batch, channel, x_dim, y_dim).
            output: Conv layer with added coordinate channels, shape (batch, channel+(2 or 3), x_dim, y_dim).
        """
        batch, _, height, width = x.size()

        # coord calculate
        xx_channel = torch.linspace(-1, 1, height, device=x.device).repeat(batch, width, 1).transpose(1, 2)
        yy_channel = torch.linspace(-1, 1, width, device=x.device).repeat(batch, height, 1)

        coords_added = torch.cat([x, xx_channel.unsqueeze(1), yy_channel.unsqueeze(1)], dim=1)

        if self.has_radial_dist:
            radial_dist = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2)).unsqueeze(1)
            coords_added = torch.cat([coords_added, radial_dist], dim=1)

        final_conv = self.conv(coords_added)
        return final_conv

class Encoder_Block(nn.Module):
    def __init__(self, conv_in_channel, conv_out_channel, hidden_channel=None, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.LeakyReLU(0.1), use_spect=False, use_coord=False):
        super(Encoder_Block, self).__init__()

        # Convolutional layer parameters From code
        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        ############################################

        hidden_channel = hidden_channel or conv_out_channel


        self.conv1 = self._coord_conv(conv_in_channel, hidden_channel, use_spect, use_coord, **kwargs_down)
        self.conv2 = self._coord_conv(hidden_channel, conv_out_channel, use_spect, use_coord, **kwargs_fine)

        # Sequential model
        self.model = nn.Sequential(norm_layer(conv_in_channel),
                                   activation_layer,
                                   self.conv1,
                                   norm_layer(hidden_channel),
                                   activation_layer,
                                   self.conv2)

    def _coord_conv(self, in_channels, out_channels, use_spect, use_coord, **kwargs):
        """
            Helper function to create a CoordConv or Conv2d layer with optional spectral normalization.
        """
        if use_coord:
            conv_layer = Coord_Conv(in_channels, out_channels, use_spect=use_spect, **kwargs)
        elif use_spect:
            conv_layer = spectral_norm(nn.Conv2d(in_channels, out_channels, **kwargs))
        else:
            conv_layer = nn.Conv2d(in_channels, out_channels, **kwargs)
        return conv_layer

    def forward(self, x):
        x = x
        x = self.model(x)
        return x

class Gated_Conv(nn.Module):
    """
        Gated Convolution Layer.
        Combines a standard convolution with a gating mechanism, where the gating mask is
        learned through a separate convolution. The output is modulated by the gating mask
        before being batch normalized.
    """

    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1,
                 norm_layer=functools.partial(nn.InstanceNorm2d, affine=True), activation=nn.LeakyReLU(0.2, inplace=True)):
        super(Gated_Conv, self).__init__()

        # Convolutional layer for feature transformation
        self.conv_feature = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=padding)

        # Convolutional layer for gating mask
        self.conv_gate = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding)

        # Batch normalization layer
        self.batch_norm = norm_layer(out_dim)

        # Activation function
        self.activation = activation

        # Gating function
        self.gate = nn.Sigmoid()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
            Initialize weights for the convolutional layers using Kaiming Normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, debugger=None):
        """
            Forward pass of the gated convolution layer.
            Args:
                x: Input tensor of shape (B, C, W, H).
            Returns:
                Tensor after gated convolution and batch normalization.
        """
        feature_response = self.conv_feature(x)
        gating_mask = self.conv_gate(x)

        gated_feature = self.activation(feature_response) * self.gate(gating_mask)
        batch_normed = self.batch_norm(gated_feature)
        if debugger is not None:
            debugger['gated_feature_response'] = feature_response
            debugger['gated_gating_mask'] = gating_mask
            debugger['gated_gated_feature'] = gated_feature
            debugger['gated_batch_normed'] = batch_normed

        return batch_normed

class Vgg_Encoder(torch.nn.Module):
    def __init__(self, pretrained_path='vgg19-dcbb9e9d.pth'):
        super(Vgg_Encoder, self).__init__()

        # Check if the pretrained model path exists
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError("The pretrained VGG model was not found at the specified path.")

        # Initialize vgg19 without pretrained weights and load from specified path
        vgg19 = torchvision.models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load(pretrained_path))
        self.features = vgg19.features

        # Freeze parameters, no gradient required
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Define which layers to return
        layers = {'10': 'conv3_1'}
        # Extract features from the image
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                break  # Exit loop early since we only want conv3_1
        return features['conv3_1']

class Res_Block(nn.Module):
    """
    A residual block that can optionally include spectral normalization, coordinate convolution,
    a learnable shortcut, and various normalization and nonlinearity layers.
    """
    def __init__(self, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d,
                 activation=nn.LeakyReLU(0.1), learnable_shortcut=False, use_spect=False): # maybe we dont need shortcut option and always use coordConv
        super(Res_Block, self).__init__()

        self.debugger = {}
        # Default values for hidden and output channels if not specified
        hidden_nc = hidden_nc or input_nc
        output_nc = output_nc or input_nc

        # Determine if a learnable shortcut is needed
        self.shortcut = learnable_shortcut or (input_nc != output_nc)

        # Convolution parameters
        conv_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv_shortcut_params = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        # Construct the main model path
        layers = [
            norm_layer(input_nc)if norm_layer is not None else nn.Identity(),
            activation,
            Coord_Conv(conv_in_channel=input_nc, conv_out_channel=hidden_nc, use_spect=use_spect, **conv_params),
            norm_layer(hidden_nc) if norm_layer is not None else nn.Identity(),
            activation,
            Coord_Conv(conv_in_channel=hidden_nc, conv_out_channel=output_nc, use_spect=use_spect, **conv_params),
        ]
        self.model = nn.Sequential(*layers)

        # Construct the shortcut path
        if self.shortcut:
            self.shortcut_path = Coord_Conv(conv_in_channel=hidden_nc, conv_out_channel=output_nc, use_spect=use_spect, **conv_shortcut_params)

    def forward(self, x, debug=False):
        shortcut = None
        model = self.model(x)
        if self.shortcut:
            shortcut = self.shortcut_path(x)
            out = model + shortcut
        else:
            out = model + x
        if debug:
            self.debugger["Res_Block_Input"] = x
            self.debugger["Res_Block_Output"] = out
            self.debugger["Res_Block_model"] = model
            self.debugger["Res_Block_Shortcut"] = shortcut
        return out


class Res_Block_Decoder(nn.Module):
    """
        Decoder block with optional spectral normalization and configurable normalization
        and non-linearity layers. Supports both Conv2d and ConvTranspose2d layers.
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True),
                 activation=nn.LeakyReLU(0.1), use_spect=False):
        super(Res_Block_Decoder, self).__init__()

        hidden_nc = hidden_nc or input_nc

        conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1)
        conv2 = nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)

        if use_spect:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)

        layers = [
            norm_layer(input_nc),
            activation,
            conv1,
            norm_layer(hidden_nc),
            activation,
            conv2,
        ]

        self.model = nn.Sequential(*layers)

        shortcut = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_spect:
            shortcut = spectral_norm(shortcut)

        # Shortcut connection with optional spectral normalization
        self.shortcut = nn.Sequential(shortcut)

    def forward(self, x):
        return self.model(x) + self.shortcut(x)

class Encoder_1(nn.Module):
    """
        Hard Encoder with configurable normalization, activation, and spectral normalization.
        Uses EncoderBlocks and ResBlockDecoders for encoding, and Gated Convolutions for feature modulation.
    """
    def __init__(self, input_nc, generator_filter_num=64, norm_layer=nn.BatchNorm2d,
                 activation=nn.LeakyReLU(0.1), use_spect=True):
        super(Encoder_1, self).__init__()

        # Define encoder blocks
        # self.encoder_blocks = nn.Sequential(
        #     Encoder_Block(input_nc, generator_filter_num*2, generator_filter_num, norm_layer, activation, use_spect),
        #     Encoder_Block(generator_filter_num*2, generator_filter_num*4, generator_filter_num*4, norm_layer, activation, use_spect),
        #     Encoder_Block(generator_filter_num*4, generator_filter_num*8, generator_filter_num*8, norm_layer, activation, use_spect),
        #     Encoder_Block(generator_filter_num*8, generator_filter_num*16, generator_filter_num*16, norm_layer, activation, use_spect)
        # )
        self.block1 = Encoder_Block(input_nc, generator_filter_num*2, generator_filter_num, norm_layer, activation, use_spect)
        self.block2 = Encoder_Block(generator_filter_num*2, generator_filter_num*4, generator_filter_num*4, norm_layer, activation, use_spect)
        self.block3 = Encoder_Block(generator_filter_num*4, generator_filter_num*8, generator_filter_num*8, norm_layer, activation, use_spect)
        self.block4 = Encoder_Block(generator_filter_num*8, generator_filter_num*16, generator_filter_num*16, norm_layer, activation, use_spect)
    def forward(self, x):
        # print("1", x)
        # x = self.encoder_blocks(x)
        # print("1", x)
        x = self.block1(x)
        # print("1", x)
        x = self.block2(x)
        # print("2", x)
        x = self.block3(x)
        # print("3", x)
        x = self.block4(x)
        # print("4", x)

        return x

class Decoder_1(nn.Module):
    """
        Hard Encoder with configurable normalization, activation, and spectral normalization.
        Uses EncoderBlocks and ResBlockDecoders for encoding, and Gated Convolutions for feature modulation.
    """
    def __init__(self, input_nc, generator_filter_num=64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True),
                 activation=nn.LeakyReLU(0.2), use_spect=True):
        super(Decoder_1, self).__init__()

        self.debugger = {}

        # Define residual blocks in the decoder
        self.res_blocks = nn.Sequential(
            Res_Block_Decoder(generator_filter_num*16, generator_filter_num*8, generator_filter_num*8, norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            Res_Block_Decoder(generator_filter_num * 8, generator_filter_num * 4, generator_filter_num * 4,norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            Res_Block_Decoder(generator_filter_num * 4, generator_filter_num * 2, generator_filter_num * 2,norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            Res_Block_Decoder(generator_filter_num * 2, generator_filter_num, generator_filter_num,norm_layer=norm_layer, activation=activation, use_spect=use_spect),
        )


    def forward(self, x, debug=False):
        out = self.res_blocks(x)
        if debug:
            self.debugger["Decoder_1_Out_Input"] = x
            self.debugger["Decoder_1_Out"] = out
        return out

class Decoder_2(nn.Module):
    def __init__(self, output_nc, ngf=64,kernel_size=3, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True),
                 activation=nn.LeakyReLU(0.1), use_spect=True):
        super(Decoder_2, self).__init__()

        self.debugger = {}
        kwargs = {'kernel_size': kernel_size, 'padding': 0, 'bias': True}
        self.model = nn.Sequential(
            Res_Block_Decoder(ngf*4, ngf*2, ngf*4, norm_layer, activation, use_spect),
            Res_Block(ngf*2, ngf*2, ngf*2, norm_layer, activation, False, use_spect),
            Res_Block_Decoder(ngf*2, ngf, ngf*2, norm_layer, activation, use_spect),
            Res_Block(ngf, ngf, ngf, norm_layer, activation, False, use_spect),
            norm_layer(ngf),
            activation,
            nn.ReflectionPad2d(int((kernel_size - 1) / 2)),
            Coord_Conv(ngf, output_nc, use_spect=use_spect, **kwargs),
            nn.Tanh()
        )

    def forward(self, input, debug=False):
        out = self.model(input)
        if debug:
            self.debugger["Decoder_2_Out_Input"] = input
            self.debugger["Decoder_2_Out"] = out
            self.debugger["Decoder_2_Out_Shape"] = out.shape
        return out


class EFB(nn.Module):
    ##extract feature block##
    def __init__(self, fin, style_length=256, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm = norm_layer(fin)
        self.debugger = {}
        norm_nc = fin
        self.style_length = style_length
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.Spade = SPADE(fin, 10)
        # to do     for unexisted seg in condition image, use mlp or conv to predict the seg class in generated image
        #        self.predict = nn.Conv2d(512,
        #        self.predict = nn.Conv2d(fin, fin, kernel_size=3, padding=1)
        param_free_norm_type = 'instance'
        ks = int(3)
        pw = ks // 2

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        self.create_gamma_beta_fc_layers()

        self.conv_gamma = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, style_codes, exist_codes, debug=False):
        #        print('ebf x: ', x.shape)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        norm1 = self.norm(x)
        [b_size, f_size, h_size, w_size] = norm1.shape
        #        print('style_codes shape', style_codes.shape)
        #        print(segmap.shape)
        middle_avg = torch.zeros((b_size, self.style_length, h_size, w_size), device=norm1.device)
        for i in range(b_size):
            for j in range(segmap.shape[1]):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    if exist_codes[i][j] == 1:
                        # print(style_codes[i][j].shape)
                        middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_codes[i][j]))

                        component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length,
                                                                                      component_mask_area)

                        middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)
                    else:
                        middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_codes[i][segmap.shape[1]]))
                        component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length,
                                                                                      component_mask_area)

                        middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)
                else:
                    middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_codes[i].mean(0, keepdim=False)))
                    component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length,
                                                                                  component_mask_area)
                    middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)

        gamma_avg = self.conv_gamma(middle_avg)
        beta_avg = self.conv_beta(middle_avg)

        # gamma_spade, beta_spade = self.Spade(segmap)

        # gamma_alpha = F.sigmoid(self.blending_gamma)
        # beta_alpha = F.sigmoid(self.blending_beta)

        gamma_final = gamma_avg  # + (1 - gamma_alpha) * gamma_spade
        beta_final = beta_avg  # + (1 - beta_alpha) * beta_spade
        out = norm1 * (1 + gamma_final) + beta_final

        return out

    def create_gamma_beta_fc_layers(self):

        ###################  These codes should be replaced with torch.nn.ModuleList
        ###################  replaced by conv 1d

        style_length = self.style_length

        self.fc_mu0 = nn.Linear(style_length, style_length)
        self.fc_mu1 = nn.Linear(style_length, style_length)
        self.fc_mu2 = nn.Linear(style_length, style_length)
        self.fc_mu3 = nn.Linear(style_length, style_length)
        self.fc_mu4 = nn.Linear(style_length, style_length)
        self.fc_mu5 = nn.Linear(style_length, style_length)
        self.fc_mu6 = nn.Linear(style_length, style_length)
        self.fc_mu7 = nn.Linear(style_length, style_length)
        # self.fc_mu8 = nn.Linear(style_length, style_length)
        # self.fc_mu9 = nn.Linear(style_length, style_length)


class Encoder_2(nn.Module):
    """
        Hard Encoder with configurable normalization, activation, and spectral normalization.
        Uses EncoderBlocks and ResBlockDecoders for encoding, and Gated Convolutions for feature modulation.
    """
    def __init__(self, input_nc, generator_filter_num=64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True), shortcut=nn.Identity,
                 activation=nn.LeakyReLU(0.1), use_spect=True):
        super(Encoder_2, self).__init__()

        self.debugger = {}
        # Define encoder blocks
        self.encoder_blocks = nn.Sequential(
            Encoder_Block(input_nc, generator_filter_num*2, None, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*2, generator_filter_num*4, None, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*4, generator_filter_num*4, None, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*4, generator_filter_num*4, None, norm_layer, activation, use_spect)
        )

        # Define residual blocks in the decoder
        self.res_blocks = nn.Sequential(
            Res_Block_Decoder(generator_filter_num*4, generator_filter_num*4, generator_filter_num*4, norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            Res_Block_Decoder(generator_filter_num*4, generator_filter_num*4, generator_filter_num*4, norm_layer=norm_layer, activation=activation, use_spect=use_spect)
        )

        # Define gated convolutions
        self.gated_convs = nn.Sequential(
            Gated_Conv(generator_filter_num*4, generator_filter_num*4),
            Gated_Conv(generator_filter_num*4, generator_filter_num*4)
        )

    def forward(self, input, debug=False):
        encoder_blocks = self.encoder_blocks(input)

        for gate in self.gated_convs:
            encoder_blocks = gate(encoder_blocks, self.debugger)
        # gated_convs = self.gated_convs(encoder_1)
        gated_convs = encoder_blocks
        # gated_convs = self.gated_convs(encoder_blocks)
        res_blocks = self.res_blocks(gated_convs)

        if debug:
            self.debugger["Encoder_2_Input"] = input
            self.debugger["Encoder_2_Encoder_Blocks"] = encoder_blocks
            self.debugger["Encoder_2_Gated_Convs"] = gated_convs
            self.debugger["Encoder_2_Out"] = res_blocks
            self.debugger["Encoder_2_Out_Shape"] = res_blocks.shape

        return res_blocks

class Encoder_3(nn.Module):
    """
        Hard Encoder with configurable normalization, activation, and spectral normalization.
        Uses EncoderBlocks and ResBlockDecoders for encoding, and Gated Convolutions for feature modulation.
    """
    def __init__(self, input_nc, generator_filter_num=64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True),
                 activation=nn.LeakyReLU(0.1), use_spect=True):
        super(Encoder_3, self).__init__()

        self.debugger = {}
        # Define encoder blocks
        self.encoder_blocks = nn.Sequential(
            Encoder_Block(input_nc, generator_filter_num*2, None, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*2, generator_filter_num*4, None, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*4, generator_filter_num*4, None, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*4, generator_filter_num*4, None, norm_layer, activation, use_spect)
        )

    def forward(self, x, debug=False):
        out = self.encoder_blocks(x)
        if debug:
            self.debugger["Encoder_3_Out"] = out
            self.debugger["Encoder_3_Out_Shape"] = out.shape
            self.debugger["Encoder_3_Input"] = x
        return out

# TODO
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class Per_Region_Encoding(nn.Module):
    """
        Per-Region Encoding with configurable normalization, activation, and spectral normalization.
    """
    def __init__(self, generator_filter_num=64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True),
                 activation=nn.LeakyReLU(0.1), use_spect=True):
        super(Per_Region_Encoding, self).__init__()

        self.debugger = {}
        # Define residual blocks in the decoder
        self.blocks = nn.Sequential(
            Res_Block_Decoder(generator_filter_num * 4, generator_filter_num * 4, generator_filter_num * 4,
                              norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            # Res_Block_Decoder(generator_filter_num * 4, generator_filter_num * 4, generator_filter_num * 4,
            #                   norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            Res_Block_Decoder(generator_filter_num * 4, generator_filter_num * 4, generator_filter_num * 4,
                              norm_layer=norm_layer, activation=activation, use_spect=use_spect),
        )

        # TODO
        self.get_code = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, padding=0), nn.Tanh())


    def forward(self,input, segmentation, debug=False):
        x = self.blocks(input)
        codes = self.get_code(x)

        segmentation_map = F.interpolate(segmentation, size=codes.size()[2:], mode='nearest')

        bs, cs, hs, ws = codes.shape
        s_size = segmentation_map.shape[1]
        codes_vector = torch.zeros((bs, s_size + 1, cs), dtype=codes.dtype, device=codes.device)
        exist_vector = torch.zeros((bs, s_size), dtype=codes.dtype, device=codes.device)
        for i in range(bs):
            for j in range(s_size):
                component_mask_area = torch.sum(segmentation_map.bool()[i, j])
                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmentation_map.bool()[i, j]).reshape(cs,
                                                                                                component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature
                    exist_vector[i][j] = 1

            # feat = x[i].reshape(1, cs, hs, ws)
            # feat_mean = feat.view(1, cs, -1).mean(dim=2).view(1, cs, 1, 1) TODO
            tmpmean, tmpstd = calc_mean_std(codes[i].reshape(1, codes[i].shape[0], codes[i].shape[1], codes[i].shape[2]))

            codes_vector[i][s_size] = tmpmean.squeeze()

        if debug:
            self.debugger["Per_Region_Encoding_In"] = input
            self.debugger["Per_Region_Encoding_Out"] = x
            self.debugger["Per_Region_Encoding_Out_Shape"] = x.shape
            self.debugger["Per_Region_Encoding_Codes_Vector"] = codes_vector
            self.debugger["Per_Region_Encoding_Exist_Vector"] = exist_vector

        return codes_vector, exist_vector, x

class Per_Region_Normalization(nn.Module):
    """
    This class implements a feature extraction block that applies normalization
    and conditional style-based modulation to an input feature map based on segmentation
    maps and style codes.
    """
    def __init__(self, input_channels, style_length=256, kernel_size=3,  norm_layer=nn.BatchNorm2d):
        super(Per_Region_Normalization, self).__init__()
        self.debugger = {}
        self.norm = norm_layer(input_channels)
        self.style_length = style_length
        self.conv_gamma = nn.Conv2d(style_length, input_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv_beta = nn.Conv2d(style_length, input_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.fc_mu_layers = nn.ModuleList([nn.Linear(style_length, style_length) for _ in range(8)]) # TODO We can use 1D convolutions instead of linear layers as well!

    def forward(self, fp, sg, style_codes, mask_codes, debug=False): #style code is per region encoding output(P(sj)
        """Applies normalization and conditional style modulation to the input features."""
        sg = F.interpolate(sg, size=fp.size()[2:], mode='nearest') # resize sg to match the input feature map
        normalized_features = self.norm(fp)
        b_size, _, h_size, w_size = normalized_features.shape
        middle_avg = torch.zeros((b_size, self.style_length, h_size, w_size), device=normalized_features.device)

        for i in range(b_size):
            for j in range(sg.shape[1]):
                component_mask = sg.bool()[i, j]
                component_mask_area = torch.sum(component_mask)
                if component_mask_area > 0:
                    style_code_idx = j if mask_codes[i][j] == 1 else sg.shape[1]
                    middle_mu = F.relu(self.fc_mu_layers[j](style_codes[i][style_code_idx]))
                    component_mu = middle_mu.view(self.style_length, 1).expand(-1, component_mask_area)
                    middle_avg[i].masked_scatter_(component_mask, component_mu)
                else: # gpt suggested remove the else! wonder why
                    middle_mu = F.relu(self.fc_mu_layers[j](style_codes[i].mean(0,keepdim=False)))
                    component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length, component_mask_area)
                    middle_avg[i].masked_scatter_(sg.bool()[i, j], component_mu)

        gamma_avg = self.conv_gamma(middle_avg)
        beta_avg = self.conv_beta(middle_avg)
        out = normalized_features * (1 + gamma_avg) + beta_avg

        if debug:
            self.debugger["Per_Region_Normalization_fp"] = fp
            self.debugger["Per_Region_Normalization_Segmentation"] = sg
            self.debugger["Per_Region_Normalization_Style_Codes"] = style_codes
            self.debugger["Per_Region_Normalization_Mask_Codes"] = mask_codes
            self.debugger["Per_Region_Normalization_Middle_Avg"] = middle_avg
            self.debugger["Per_Region_Normalization_Gamma_Avg"] = gamma_avg
            self.debugger["Per_Region_Normalization_Beta_Avg"] = beta_avg
            self.debugger["Per_Region_Normalization_Out"] = out
            self.debugger["Per_Region_Normalization_Out_Shape"] = out.shape

        return out

class Res_Block_Encoder(nn.Module):
    """
    Residual Block for Encoder
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU(0.1), use_spect=False):
        super(Res_Block_Encoder, self).__init__()

        hidden_nc = hidden_nc or input_nc

        conv1 = nn.Conv2d(input_nc, hidden_nc, 3, stride=1, padding=1)
        conv2 = nn.Conv2d(hidden_nc, output_nc, 4, stride=2, padding=1)

        if use_spect:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)

        layers = [
            conv1,
            activation,
            conv2,
        ]

        if norm_layer is not None:
            layers.insert(1, norm_layer(hidden_nc))
            layers.append(norm_layer(output_nc))

        # Shortcut to match dimensions and add bypass
        shortcut = [
            nn.AvgPool2d(2, stride=2),
            spectral_norm(nn.Conv2d(input_nc, output_nc, 1, stride=1, padding=0))
        ]

        self.model = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        return self.model(x) + self.shortcut(x)


import sys
import re
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic feature map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segfeature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segfeature = F.interpolate(segfeature, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segfeature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

    def hook(self, x, segfeature):
        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(segfeature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out, gamma, beta


class GetMatrix(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GetMatrix, self).__init__()
        self.get_gamma = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.get_beta = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        gamma = self.get_gamma(x)
        beta = self.get_beta(x)
        return gamma, beta


class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        use_bias = True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)

    def forward(self, x, feature):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1, 1)
        beta = beta.view(*beta.size()[:2], 1, 1)
        out = normalized * (1 + gamma) + beta

        return out


class VggEncoder(nn.Module):
    def __init__(self):
        super(VggEncoder, self).__init__()
        # self.vgg = models.vgg19(pretrained=True).features
        vgg19 = torchvision.models.vgg.vgg19(pretrained=False)
        # You can download vgg19-dcbb9e9d.pth from https://github.com/pytorch/vision/tree/master/torchvision/models and change the related path.
        vgg19.load_state_dict(torch.load('./vgg19-dcbb9e9d.pth'))
        self.vgg = vgg19.features

        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, model, layers=None):
        if layers is None:
            layers = {'10': 'conv3_1'}
        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, x):

        sty_fea = self.get_features(x, self.vgg)
        return sty_fea['conv3_1']


def get_norm_layer(norm_type='batch'):
    """Get the normalization layer for the networks"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'adain':
        norm_layer = functools.partial(ADAIN)
    elif norm_type == 'spade':
        norm_layer = functools.partial(SPADE, config_text='spadeinstance3x3')
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    if norm_type != 'none':
        norm_layer.__name__ = norm_type

    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + 1 + opt.iter_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f M' % (num_params / 1e6))


def init_net(net, init_type='normal', activation='relu', gpu_ids=[]):
    """print the network structure and initial the network"""
    print_network(net)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module


def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, **kwargs):
    """use coord convolution layer to add position information"""
    if use_coord:
        return CoordConv(input_nc, output_nc, with_r, use_spect, **kwargs)
    else:
        return spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)


######################################################################################
# Network basic function
######################################################################################
class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """

    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    CoordConv operation
    """

    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module


class EncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(EncoderBlock, self).__init__()

        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_down)
        conv2 = coord_conv(output_nc, output_nc, use_spect, use_coord, **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(output_nc), nonlinearity, conv2, )

    def forward(self, x):
        out = self.model(x)
        return out


class BlockEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False):
        super(BlockEncoder, self).__init__()

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=4, stride=2, padding=1), use_spect)
        conv2 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=3, stride=1, padding=1), use_spect)
        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(output_nc), nonlinearity, conv2, )

    def forward(self, x):
        out = self.model(x)
        return out


class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """

    def __init__(self, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 learnable_shortcut=False, use_spect=False, use_coord=False):
        super(ResBlock, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc
        self.learnable_shortcut = True if input_nc != output_nc else learnable_shortcut

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(hidden_nc), nonlinearity, conv2, )

        if self.learnable_shortcut:
            bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)
            self.shortcut = nn.Sequential(bypass, )

    def forward(self, x):
        if self.learnable_shortcut:
            out = self.model(x) + self.shortcut(x)
        else:
            out = self.model(x) + x
        return out


class ResBlocks(nn.Module):
    """docstring for ResBlocks"""

    def __init__(self, num_blocks, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d,
                 nonlinearity=nn.LeakyReLU(), learnable_shortcut=False, use_spect=False, use_coord=False):
        super(ResBlocks, self).__init__()
        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc

        self.model = []
        if num_blocks == 1:
            self.model += [ResBlock(input_nc, output_nc, hidden_nc,
                                    norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]

        else:
            self.model += [ResBlock(input_nc, hidden_nc, hidden_nc,
                                    norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]
            for i in range(num_blocks - 2):
                self.model += [ResBlock(hidden_nc, hidden_nc, hidden_nc,
                                        norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]
            self.model += [ResBlock(hidden_nc, output_nc, hidden_nc,
                                    norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]

        self.model = nn.Sequential(*self.model)

    def forward(self, inputs):
        return self.model(inputs)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class Zencoder(nn.Module):
    """ extract style amtrix """

    def __init__(self, input_nc, ngf=64, norm='instance', act='LeakyReLU', use_spect=True):
        super(Zencoder, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        acti = get_nonlinearity_layer(activation_type=act)

        self.block0 = EncoderBlock(input_nc, ngf * 2, norm_layer, acti, use_spect)
        self.block1 = EncoderBlock(ngf * 2, ngf * 4, norm_layer, acti, use_spect)
        self.block2 = EncoderBlock(ngf * 4, ngf * 4, norm_layer, acti, use_spect)
        self.block3 = EncoderBlock(ngf * 4, ngf * 4, norm_layer, acti, use_spect)
        self.block4 = ResBlockDecoder(ngf * 4, ngf * 4, ngf * 4, norm_layer, acti, use_spect)
        self.block5 = ResBlockDecoder(ngf * 4, ngf * 4, ngf * 4, norm_layer, acti, use_spect)

        self.down = nn.Upsample(scale_factor=0.25, mode='nearest')
        self.get_code = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, padding=0), nn.Tanh())

    def forward(self, input, seg):
        out = self.block0(input)
        out = self.block3(self.block2(self.block1(out)))
        out = self.block5(self.block4(out))

        codes = self.get_code(out)

        segmap = F.interpolate(seg, size=codes.size()[2:], mode='nearest')

        bs = codes.shape[0]
        hs = codes.shape[2]
        ws = codes.shape[3]
        cs = codes.shape[1]
        f_size = cs

        s_size = segmap.shape[1]

        codes_vector = torch.zeros((bs, s_size + 1, cs), dtype=codes.dtype, device=codes.device)
        exist_vector = torch.zeros((bs, s_size), dtype=codes.dtype, device=codes.device)

        for i in range(bs):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])
                #             tmpcom = torch.zeros((f_size, h_size, w_size), dtype=codes.dtype, device=codes.device)

                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,
                                                                                                  component_mask_area).mean(
                        1)

                    #                 t = codes_component_feature.masked_scatter_(segmap.bool()[i, j],tmpcom )
                    #                 print(t.shape)
                    codes_vector[i][j] = codes_component_feature
                    exist_vector[i][j] = 1
                    # codes_avg[i].masked_scatter_(segmap.bool()[i, j], codes_component_mu)
            tmpmean, tmpstd = calc_mean_std(
                codes[i].reshape(1, codes[i].shape[0], codes[i].shape[1], codes[i].shape[2]))
            codes_vector[i][s_size] = tmpmean.squeeze()

        return codes_vector, exist_vector, out


class BasicEncoder(nn.Module):
    """ extract style amtrix """

    def __init__(self, input_nc, ngf=64, norm='instance', act='LeakyReLU', use_spect=True):
        super(BasicEncoder, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        acti = get_nonlinearity_layer(activation_type=act)

        self.block0 = EncoderBlock(input_nc, ngf * 2, norm_layer, acti, use_spect)
        self.block1 = EncoderBlock(ngf * 2, ngf * 4, norm_layer, acti, use_spect)

        # self.get_code = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(128, 256, kernel_size=3, padding=0), nn.Tanh())

    def forward(self, input):
        out = self.block0(input)
        out = self.block1(out)
        return out


class HardEncoder(nn.Module):
    """ hard encoder """

    def __init__(self, input_nc, ngf=64, norm='instance', act='LeakyReLU', use_spect=True):
        super(HardEncoder, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm)
        acti = get_nonlinearity_layer(activation_type=act)

        self.block0 = EncoderBlock(input_nc, ngf * 2, norm_layer, acti, use_spect)
        self.block1 = EncoderBlock(ngf * 2, ngf * 4, norm_layer, acti, use_spect)
        self.block2 = EncoderBlock(ngf * 4, ngf * 4, norm_layer, acti, use_spect)
        self.block3 = EncoderBlock(ngf * 4, ngf * 4, norm_layer, acti, use_spect)

        self.block4 = ResBlockDecoder(ngf * 4, ngf * 4, ngf * 4, norm_layer, acti, use_spect)
        self.block5 = ResBlockDecoder(ngf * 4, ngf * 4, ngf * 4, norm_layer, acti, use_spect)

        self.deform3 = Gated_conv(ngf * 4, ngf * 4)
        self.deform4 = Gated_conv(ngf * 4, ngf * 4)

    def forward(self, input):
        out = self.block0(input)
        out = self.block3(self.block2(self.block1(out)))
        out = self.deform4(self.deform3(out))
        out = self.block5(self.block4(out))
        return out


class BasicDecoder(nn.Module):

    def __init__(self, output_nc, ngf=64, norm='instance', act='LeakyReLU', use_spect=True):
        super(BasicDecoder, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm)
        acti = get_nonlinearity_layer(activation_type=act)

        self.block0 = ResBlockDecoder(ngf * 4, ngf * 2, ngf * 4, norm_layer, acti, use_spect)
        self.block1 = ResBlock(ngf * 2, output_nc=ngf * 2, hidden_nc=ngf * 2, norm_layer=norm_layer, nonlinearity=acti,
                               learnable_shortcut=False, use_spect=True, use_coord=False)

        self.block2 = ResBlockDecoder(ngf * 2, ngf, ngf * 2, norm_layer, acti, use_spect)
        self.block3 = ResBlock(ngf, output_nc=ngf, hidden_nc=ngf, norm_layer=norm_layer, nonlinearity=acti,
                               learnable_shortcut=False, use_spect=True, use_coord=False)

        self.out = Output(ngf, output_nc, 3, norm_layer, acti)

    def forward(self, input):
        x = self.block1(self.block0(input))
        x = self.block3(self.block2(x))
        x = self.out(x)

        return (x)


class EFB(nn.Module):
    ##extract feature block##
    def __init__(self, fin, style_length=256, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm = norm_layer(fin)

        norm_nc = fin
        self.style_length = style_length
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.Spade = SPADE(fin, 10)
        # to do     for unexisted seg in condition image, use mlp or conv to predict the seg class in generated image
        #        self.predict = nn.Conv2d(512,
        #        self.predict = nn.Conv2d(fin, fin, kernel_size=3, padding=1)
        param_free_norm_type = 'instance'
        ks = int(3)
        pw = ks // 2

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        self.create_gamma_beta_fc_layers()

        self.conv_gamma = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, style_codes, exist_codes):
        #        print('ebf x: ', x.shape)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        norm1 = self.norm(x)
        [b_size, f_size, h_size, w_size] = norm1.shape
        #        print('style_codes shape', style_codes.shape)
        #        print(segmap.shape)
        middle_avg = torch.zeros((b_size, self.style_length, h_size, w_size), device=norm1.device)
        for i in range(b_size):
            for j in range(segmap.shape[1]):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    if exist_codes[i][j] == 1:
                        # print(style_codes[i][j].shape)
                        middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_codes[i][j]))

                        component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length,
                                                                                      component_mask_area)

                        middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)
                    else:
                        middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_codes[i][segmap.shape[1]]))
                        component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length,
                                                                                      component_mask_area)

                        middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)
                else:
                    middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_codes[i].mean(0, keepdim=False)))
                    component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length,
                                                                                  component_mask_area)
                    middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)

        gamma_avg = self.conv_gamma(middle_avg)
        beta_avg = self.conv_beta(middle_avg)

        # gamma_spade, beta_spade = self.Spade(segmap)

        # gamma_alpha = F.sigmoid(self.blending_gamma)
        # beta_alpha = F.sigmoid(self.blending_beta)

        gamma_final = gamma_avg  # + (1 - gamma_alpha) * gamma_spade
        beta_final = beta_avg  # + (1 - beta_alpha) * beta_spade
        out = norm1 * (1 + gamma_final) + beta_final

        return out

    def create_gamma_beta_fc_layers(self):

        ###################  These codes should be replaced with torch.nn.ModuleList
        ###################  replaced by conv 1d

        style_length = self.style_length

        self.fc_mu0 = nn.Linear(style_length, style_length)
        self.fc_mu1 = nn.Linear(style_length, style_length)
        self.fc_mu2 = nn.Linear(style_length, style_length)
        self.fc_mu3 = nn.Linear(style_length, style_length)
        self.fc_mu4 = nn.Linear(style_length, style_length)
        self.fc_mu5 = nn.Linear(style_length, style_length)
        self.fc_mu6 = nn.Linear(style_length, style_length)
        self.fc_mu7 = nn.Linear(style_length, style_length)
        # self.fc_mu8 = nn.Linear(style_length, style_length)
        # self.fc_mu9 = nn.Linear(style_length, style_length)


class Gated_conv(nn.Module):
    """ Gated convlution Layer"""

    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, dilation=1, \
                 groups=1, bias=True, norm_layer=nn.InstanceNorm2d, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(Gated_conv, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.gated_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation)
        self.mask_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation)
        self.batch_norm2d = norm_layer(out_dim)
        self.sigmoid = nn.Sigmoid()  #

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.kaiming_normal(m.weight_bar)
                else:
                    nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        res = x
        x = self.gated_conv(x)
        mask = self.mask_conv(res)

        if self.activation is not None:
            x = self.activation(x) * self.sigmoid(mask)
        else:
            x = x * self.sigmoid(mask)
        return self.batch_norm2d(x)


class SelfAttentionBlock(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttentionBlock, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class GlobalAttentionBlock(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(GlobalAttentionBlock, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, source, target):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = source.size()
        proj_query = self.query_conv(source).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(target).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(source).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + target
        return out, attention


class BilinearSamplingBlock(nn.Module):
    """docstring for BilinearSamplingBlock"""

    def __init__(self):
        super(BilinearSamplingBlock, self).__init__()

    def forward(self, source, flow_field):
        [b, _, h, w] = source.size()
        # flow_field = torch.nn.functional.interpolate(flow_field, (w,h))
        x = torch.arange(w).view(1, -1).expand(h, -1)
        y = torch.arange(h).view(-1, 1).expand(-1, w)
        grid = torch.stack([x, y], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2 * grid / (w - 1) - 1
        flow_field = 2 * flow_field / (w - 1)
        grid = (grid + flow_field).permute(0, 2, 3, 1)
        warp = torch.nn.functional.grid_sample(source, grid)
        return warp


class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """

    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(
            nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        bypass = spectral_norm(
            nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, norm_layer(hidden_nc), nonlinearity,
                                       conv2, )

        self.shortcut = nn.Sequential(bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out


class ResBlockEncoder(nn.Module):
    """
    Define a decoder block
    """

    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockEncoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1), use_spect)
        bypass = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(hidden_nc), nonlinearity, conv2, )
        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out

    # ResNet block that uses SPADE.


# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    # def __init__(self, fin, fout, opt):
    #     super().__init__()

    def __init__(self, input_nc, output_nc, hidden_nc, label_nc, spade_config_str='spadeinstance3x3',
                 nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False, learned_shortcut=False):
        super(SPADEResnetBlock, self).__init__()
        # Attributes
        self.learned_shortcut = (input_nc != output_nc) or learned_shortcut
        self.actvn = nonlinearity
        hidden_nc = min(input_nc, output_nc) if hidden_nc is None else hidden_nc

        # create conv layers
        self.conv_0 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        self.conv_1 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=3, stride=1, padding=1), use_spect)
        # self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False), use_spect)

        # define normalization layers
        self.norm_0 = SPADE(spade_config_str, input_nc, label_nc)
        self.norm_1 = SPADE(spade_config_str, hidden_nc, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, input_nc, label_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s


class ADAINResnetBlock(nn.Module):
    # def __init__(self, fin, fout, opt):
    #     super().__init__()

    def __init__(self, input_nc, output_nc, hidden_nc, feature_nc, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False, learned_shortcut=False):
        super(ADAINResnetBlock, self).__init__()
        # Attributes
        self.learned_shortcut = (input_nc != output_nc) or learned_shortcut
        self.actvn = nonlinearity
        hidden_nc = min(input_nc, output_nc) if hidden_nc is None else hidden_nc

        # create conv layers
        self.conv_0 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        self.conv_1 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=3, stride=1, padding=1), use_spect)
        # self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False), use_spect)

        # define normalization layers
        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(hidden_nc, feature_nc)
        if self.learned_shortcut:
            self.norm_s = ADAIN(input_nc, feature_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, z):
        x_s = self.shortcut(x, z)
        dx = self.conv_0(self.actvn(self.norm_0(x, z)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, z)))
        out = x_s + dx
        return out

    def shortcut(self, x, z):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, z))
        else:
            x_s = x
        return x_s


class Output(nn.Module):
    """
    Define the output layer
    """

    def __init__(self, input_nc, output_nc, kernel_size=3, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=True, use_coord=False):
        super(Output, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding': 0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)),
                                       self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out


class Jump(nn.Module):
    """
    Define the output layer
    """

    def __init__(self, input_nc, output_nc, kernel_size=3, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Jump, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding': 0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)),
                                       self.conv1)

    def forward(self, x):
        out = self.model(x)
        return out


class LinearBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False):
        super(LinearBlock, self).__init__()
        use_bias = True

        self.fc = spectral_norm(nn.Linear(input_nc, output_nc, bias=use_bias), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.fc)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.fc)

    def forward(self, x):
        out = self.model(x)
        return out


class LayerNorm1d(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm1d, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape),
                                self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)


class ADALN1d(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()
        nhidden = 128
        use_bias = True
        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)

    def forward(self, x, feature):
        normalized_shape = x.size()[1:]

        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        gamma = gamma.view(*gamma.size()[:2], 1)
        beta = beta.view(*beta.size()[:2], 1)
        out = F.layer_norm(x, normalized_shape) * (1 + gamma) + beta

        return out
