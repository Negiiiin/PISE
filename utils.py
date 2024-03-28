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
                 activation_layer=nn.LeakyReLU(), use_spect=False, use_coord=False):
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
        conv_layer = Coord_Conv(in_channels, out_channels, use_spect=use_spect, **kwargs) if use_coord \
                     else nn.Conv2d(in_channels, out_channels, **kwargs)
        return conv_layer

    def forward(self, x):
        return self.model(x)

class Gated_Conv(nn.Module):
    """
        Gated Convolution Layer.
        Combines a standard convolution with a gating mechanism, where the gating mask is
        learned through a separate convolution. The output is modulated by the gating mask
        before being batch normalized.
    """

    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1,
                 norm_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU(0.2, inplace=True)):
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

    def forward(self, x):
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

        return self.batch_norm(gated_feature)

class Vgg_Encoder(torch.nn.Module):
    def __init__(self, pretrained_path='/content/drive/MyDrive/PISE/vgg19-dcbb9e9d.pth'):
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
                 activation=nn.LeakyReLU(), learnable_shortcut=False, use_spect=False, shortcut=nn.Identity): # maybe we dont need shortcut option and always use coordConv
        super(Res_Block, self).__init__()

        # Default values for hidden and output channels if not specified
        hidden_nc = hidden_nc or input_nc
        output_nc = output_nc or input_nc

        # Determine if a learnable shortcut is needed
        self.learnable_shortcut = learnable_shortcut or (input_nc != output_nc)

        # Convolution parameters
        conv_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv_shortcut_params = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        # Construct the main model path
        layers = [
            norm_layer(input_nc),
            activation,
            Coord_Conv(conv_in_channel=input_nc, conv_out_channel=hidden_nc, use_spect=use_spect, **conv_params),
            norm_layer(hidden_nc) if norm_layer is not None else nn.Identity(),
            activation,
            Coord_Conv(conv_in_channel=hidden_nc, conv_out_channel=output_nc, use_spect=use_spect, **conv_params),
        ]
        self.model = nn.Sequential(*layers)

        # Construct the shortcut path
        self.shortcut = shortcut
        # can be nn.Sequential(
             #   Coord_Conv(input_nc, output_nc, use_spect=use_spect, **conv_shortcut_params)
            #)

    def forward(self, x):
        return self.model(x) + self.shortcut(x)

class Res_Block_Decoder(nn.Module):
    """
        Decoder block with optional spectral normalization and configurable normalization
        and non-linearity layers. Supports both Conv2d and ConvTranspose2d layers.
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d,
                 activation=nn.LeakyReLU(), use_spect=False):
        super(Res_Block_Decoder, self).__init__()

        hidden_nc = hidden_nc or input_nc

        conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1)
        conv2 = nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)

        if use_spect:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)

        layers = [
            norm_layer(input_nc),
            conv1,
            activation,
            norm_layer(hidden_nc),
            conv2
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
        self.encoder_blocks = nn.Sequential(
            Encoder_Block(input_nc, generator_filter_num*2, generator_filter_num, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*2, generator_filter_num*4, generator_filter_num*4, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*4, generator_filter_num*8, generator_filter_num*8, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*8, generator_filter_num*16, generator_filter_num*16, norm_layer, activation, use_spect)
        )
    def forward(self, x):
        x = self.encoder_blocks(x)
        return x

class Decoder_1(nn.Module):
    """
        Hard Encoder with configurable normalization, activation, and spectral normalization.
        Uses EncoderBlocks and ResBlockDecoders for encoding, and Gated Convolutions for feature modulation.
    """
    def __init__(self, input_nc, generator_filter_num=64, norm_layer=nn.BatchNorm2d,
                 activation=nn.LeakyReLU(0.1), use_spect=True):
        super(Decoder_1, self).__init__()

        # Define residual blocks in the decoder
        self.res_blocks = nn.Sequential(
            Res_Block_Decoder(generator_filter_num*16, generator_filter_num*8, generator_filter_num*8, norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            Res_Block_Decoder(generator_filter_num * 8, generator_filter_num * 4, generator_filter_num * 4,norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            Res_Block_Decoder(generator_filter_num * 4, generator_filter_num * 2, generator_filter_num * 2,norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            Res_Block_Decoder(generator_filter_num * 2, generator_filter_num, generator_filter_num,norm_layer=norm_layer, activation=activation, use_spect=use_spect),
        )


    def forward(self, x):
        x = self.res_blocks(x)
        return x





class Decoder_2(nn.Module):
    def __init__(self, output_nc, ngf=64,kernel_size=3, norm_layer=nn.BatchNorm2d,
                 activation=nn.LeakyReLU(0.1), use_spect=True):
        super(Decoder_2, self).__init__()


        kwargs = {'kernel_size': kernel_size, 'padding': 0, 'bias': True}
        self.model = nn.Sequential(
            Res_Block_Decoder(ngf*4, ngf*2, ngf*4, norm_layer, activation, use_spect),
            Res_Block(ngf*2, ngf*2, ngf*2, norm_layer, activation, False, use_spect, False),
            Res_Block_Decoder(ngf*2, ngf, ngf*2, norm_layer, activation, use_spect),
            Res_Block(ngf, ngf, ngf, norm_layer, activation, False, use_spect, False),
            norm_layer(output_nc),
            activation,
            nn.ReflectionPad2d(int((kernel_size - 1) / 2)),
            Coord_Conv(output_nc, 8, use_spect=use_spect, **kwargs),
            nn.Tanh()
        )



    def forward(self, input):
        return self.model(input)


class Parsing_Generator(nn.Module):
    """
        Hard Encoder with configurable normalization, activation, and spectral normalization.
        Uses EncoderBlocks and ResBlockDecoders for encoding, and Gated Convolutions for feature modulation.
    """
    def __init__(self, input_nc, generator_filter_num=64, norm_layer=nn.BatchNorm2d,
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

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}


        self.output = nn.Sequential(
            norm_layer(input_nc),
            activation,
            nn.ReflectionPad2d(int((kernel_size - 1) / 2)),
            Coord_Conv(input_nc, 8, use_spect=use_spect, **kwargs),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder_1(x)
        x = self.gated_convs(x)
        x = self.decoder_1(x)
        x = self.output(x)
        # x = (x + 1.) / 2. TODO test
        return x

class Encoder_2(nn.Module):
    """
        Hard Encoder with configurable normalization, activation, and spectral normalization.
        Uses EncoderBlocks and ResBlockDecoders for encoding, and Gated Convolutions for feature modulation.
    """
    def __init__(self, input_nc, generator_filter_num=64, norm_layer=nn.BatchNorm2d, shortcut=nn.Identity,
                 activation=nn.LeakyReLU(0.1), use_spect=True):
        super(Encoder_2, self).__init__()

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

    def forward(self, x):
        x = self.encoder_blocks(x)
        x = self.gated_convs(x)
        x = self.res_blocks(x)
        return x



class Encoder_3(nn.Module):
    """
        Hard Encoder with configurable normalization, activation, and spectral normalization.
        Uses EncoderBlocks and ResBlockDecoders for encoding, and Gated Convolutions for feature modulation.
    """
    def __init__(self, input_nc, generator_filter_num=64, norm_layer=nn.InstanceNorm2d,
                 activation=nn.LeakyReLU(0.1), use_spect=True):
        super(Encoder_3, self).__init__()

        # Define encoder blocks
        self.encoder_blocks = nn.Sequential(
            Encoder_Block(input_nc, generator_filter_num*2, None, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*2, generator_filter_num*4, None, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*4, generator_filter_num*4, None, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*4, generator_filter_num*4, None, norm_layer, activation, use_spect)
        )

    def forward(self, x):
        x = self.encoder_blocks(x)
        return x


class Per_Region_Encoding(nn.Module):
    """
        Per-Region Encoding with configurable normalization, activation, and spectral normalization.
    """
    def __init__(self, generator_filter_num=64, norm_layer=nn.InstanceNorm2d,
                 activation=nn.LeakyReLU(0.1), use_spect=True):
        super(Per_Region_Encoding, self).__init__()

        # Define residual blocks in the decoder
        self.blocks = nn.Sequential(
            Res_Block_Decoder(generator_filter_num * 4, generator_filter_num * 4, generator_filter_num * 4,
                              norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            Res_Block_Decoder(generator_filter_num * 4, generator_filter_num * 4, generator_filter_num * 4,
                              norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            Res_Block_Decoder(generator_filter_num * 4, generator_filter_num * 4, generator_filter_num * 4,
                              norm_layer=norm_layer, activation=activation, use_spect=use_spect),
            nn.Conv2d(256, 256, kernel_size=1, padding=0),
            nn.Tanh()
        )


    def forward(self,x, segmentation):
        x = self.blocks(x)
        segmentation_map = F.interpolate(segmentation, size=x.size()[2:], mode='nearest')

        bs, cs, hs, ws = x.shape
        s_size = segmentation_map.shape[1]
        codes_vector = torch.zeros((bs, s_size + 1, cs), dtype=x.dtype, device=x.device)
        exist_vector = torch.zeros((bs, s_size), dtype=x.dtype, device=x.device)
        for i in range(bs):
            for j in range(s_size):
                component_mask_area = torch.sum(segmentation_map.bool()[i, j])
                if component_mask_area > 0:
                    codes_component_feature = x[i].masked_select(segmentation_map.bool()[i, j]).reshape(cs,
                                                                                                component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature
                    exist_vector[i][j] = 1

            feat = x[i].reshape(1, cs, hs, ws)
            feat_mean = feat.view(1, cs, -1).mean(dim=2).view(1, cs, 1, 1)

            codes_vector[i][s_size] = feat_mean.squeeze()

        return codes_vector, exist_vector, x


class Per_Region_Normalization(nn.Module):
    """
    This class implements a feature extraction block that applies normalization
    and conditional style-based modulation to an input feature map based on segmentation
    maps and style codes.
    """
    def __init__(self, input_channels, style_length=256, kernel_size=3,  norm_layer=nn.BatchNorm2d):
        super(Per_Region_Normalization, self).__init__()
        self.norm = norm_layer(input_channels)
        self.style_length = style_length
        self.conv_gamma = nn.Conv2d(style_length, input_channels, kernel_size=kernel_size, padding=(kernel_size-1)/2)
        self.conv_beta = nn.Conv2d(style_length, input_channels, kernel_size=kernel_size, padding=(kernel_size-1)/2)
        self.fc_mu_layers = nn.ModuleList([nn.Linear(style_length, style_length) for _ in range(8)]) # TODO We can use 1D convolutions instead of linear layers as well!

    def forward(self, fp, sg, style_codes, mask_codes): #style code is per region encoding output(P(sj)
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
        return out



# ----------------------------------------------- Discriminator -----------------------------------------------


class Res_Block_Encoder(nn.Module):
    """
    Residual Block for Encoder
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d(), activation=nn.LeakyReLU(), use_spect=False):
        super(Res_Block_Encoder, self).__init__()

        hidden_nc = hidden_nc or input_nc

        conv1 = nn.Conv2d(input_nc, hidden_nc, 3, stride=1, padding=1)
        conv2 = nn.Conv2d(hidden_nc, output_nc, 4, stride=2, padding=1)

        if use_spect:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)

        layers = [
            conv1,
            norm_layer(hidden_nc),
            activation(),
            conv2,
            norm_layer(output_nc)
        ]

        # Shortcut to match dimensions and add bypass
        shortcut = [
            nn.AvgPool2d(2, stride=2),
            spectral_norm(nn.Conv2d(input_nc, output_nc, 1, stride=1, padding=0))
        ]

        self.model = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        return self.model(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm_layer=nn.Identity(), activation=nn.LeakyReLU,
                 use_spect=True):
        super(Discriminator, self).__init__()
        self.layers = layers
        self.activation = activation

        self.blocks = [Res_Block_Encoder(input_nc if i == 0 else ndf * min(2 ** (i - 1), img_f // ndf),
                                       ndf * min(2 ** i, img_f // ndf),
                                       ndf * min(2 ** (i - 1), img_f // ndf) if i > 0 else ndf,
                                       norm_layer, activation, use_spect) for i in range(layers)]

        self.blocks = nn.ModuleList(self.blocks)
        self.final_conv = spectral_norm(nn.Conv2d(ndf * min(2 ** (layers - 1), img_f // ndf), 1, 1))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_conv(self.activation(x))


class Generator(nn.Module):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64,
                     activation=nn.LeakyReLU, use_spect=True, use_coord=False):
            super(Generator, self).__init__()

            self.use_coordconv = True
            self.match_kernel = 3


            self.parsing_generator = Parsing_Generator(8 + 18 * 2, 8)

            self.encoder_3 = Encoder_3(3, ngf)  # encoder that gets image S as input
            self.per_region_encoding = Per_Region_Encoding(ngf)


            self.vgg19 = VGG19()
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

        parcode = self.parsing_generator(torch.cat((par1, pose1, pose2), 1))  # parsing utput
        par2 = parcode

        parcode = self.encoder_2(torch.cat((par1, par2, pose2, img1), 1))  # Fp


        parcode = self.per_region_normalization(parcode, par2, codes_vector, exist_vector)
        # parcode = self.res(parcode)

        ## regularization to let transformed code and target image code in the same feature space

        img2code = self.vgg19(img2)  # VGG output of original image
        loss_reg = F.mse_loss(img2code, parcode)

        parcode = self.decoder_2(parcode)
        return parcode, loss_reg, par2



class Final_Model(nn.Module):
    def __init__(self, gpu_ids, device, save_dir, lr=1e-4, ratio_g2d=0.1):
        super(Final_Model, self).__init__()
        self.gpu_ids = gpu_ids
        self.is_train = True
        self.optimizers = []
        self.device = device
        self.save_dir = save_dir

        # Define the generator
        self.generator = Generator()
        self.discriminator = Discriminator(ndf=32, img_f=128, layers=4)

        # Initialize loss functions and optimizers if training
        if self.isTrain:
            self.init_losses_and_optimizers(device, lr, ratio_g2d)

    def init_weights(self, gain=0.02):
        # Iterate over all modules in the model
        for m in self.modules():
            class_name = m.__class__.__name__
            # Initialize weights for BatchNorm2d layers
            if class_name.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            # Initialize weights for Conv and Linear layers using orthogonal initialization
            elif hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1):
                init.orthogonal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def init_losses_and_optimizers(self, device, lr, ratio_g2d):
        self.GAN_loss = loss.Adversarial_Loss().to(device)
        self.L1_loss = nn.L1Loss()
        self.VGG_loss = loss.VGG_Loss().to(device)
        self.cross_entropy_2d_loss = loss.Cross_Entropy_Loss2d()

        # Optimizer for the generator
        self.optimizer_G = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.generator.parameters()),
            lr=lr, betas=(0.9, 0.999)
        )
        self.optimizers.append(self.optimizer_G)

        # Optimizer for the discriminator
        self.optimizer_D = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.discriminator.parameters()),
            lr=lr * ratio_g2d, betas=(0.9, 0.999)
        )
        self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.input = input
        self.image_paths = []

        # Define keys for easier readability and modification
        keys = ['P1', 'BP1', 'SPL1', 'P2', 'BP2', 'SPL2']
        # Automatically transfer all tensors to the specified device
        for key in keys:
            if key in input:
                setattr(self, 'input_' + key, input[key].to(self.device))

        self.label_P2 = input['label_P2'].to(self.device)

        # Handle image paths separately as they are not tensors
        for i in range(input['P1'].size(0)):  # Assuming 'P1' exists and has a batch dimension
            path = '_'.join([os.path.splitext(input[path_key][i])[0] for path_key in ['P1_path', 'P2_path']])
            self.image_paths.append(path)

    def forward(self):
        self.generated_img, self.loss_reg, self.parsav = self.net_G(self.input_P1, self.input_P2, self.input_BP1, self.input_BP2, self.input_SPL1, self.input_SPL2) #TODO

    def backward_D(self, real_input, fake_input, unfreeze_netD=True):
        """Calculate the GAN loss for the discriminator, including handling WGAN-GP if specified."""
        if unfreeze_netD:
            # Unfreeze the discriminator if it was frozen
            for param in self.discriminator.parameters():
                param.requires_grad = True

        # Real input loss
        D_real = self.discriminator(real_input)
        D_real_loss = self.GAN_loss(D_real, True)

        # Fake input loss (detach to avoid backproping into the generator)
        D_fake = self.discriminator(fake_input.detach())
        D_fake_loss = self.GAN_loss(D_fake, False)

        # Combined discriminator loss
        D_loss = (D_real_loss + D_fake_loss) * 0.5

        # Backpropagate the discriminator loss
        D_loss.backward()

        return D_loss

    def backward_G(self, lambda_regularization=30.0, lambda_rec=5.0, lambda_pl=100.0, lambda_a=2.0, lambda_style=200.0, lambda_content=0.5):
        """Calculate training loss for the generator."""
        # Initialize total loss
        total_loss = 0

        # Parsing Generator two losses ---------------------------
        label_P2 = self.label_P2.squeeze(1).long()
        self.parsing_gen_cross = self.parLoss(self.parsav, label_P2)
        self.parsing_gen_l1 = self.L1_loss(self.parsav,
                                           self.input_SPL2) * lambda_pl  # l1 distance loss between the generated parsing map and the target parsing map
        total_loss += self.parsing_gen_cross + self.parsing_gen_l1

        # Image generator losses ---------------------------------
        # Regularization loss
        self.L_cor = self.loss_reg * lambda_regularization # self.loss_reg is an output of the generator - Lcor TODO
        total_loss += self.L_cor

        # L1 loss (Appearance loss)
        self.L_l1 = self.L1_loss(self.generated_img, self.input_P2) * lambda_rec # Ll1 - self.generated_img is an output of the generator - TODO
        total_loss += self.L_l1

        # Freeze the discriminator for the generator's backward pass
        for param in self.net_D.parameters():
            param.requires_grad = False

        # GAN loss (Adversarial loss)
        D_fake = self.discriminator(self.generated_img) #TODO
        self.loss_adv = self.GAN_loss(D_fake, True, False) * lambda_a
        total_loss += self.loss_adv

        # Perceptual loss (Content and Style)
        self.loss_content_gen, self.loss_style_gen = self.VGG_loss(self.generated_img, self.input_P2) #TODO
        self.loss_style_gen *= lambda_style
        self.loss_content_gen *= lambda_content
        total_loss += self.loss_content_gen + self.loss_style_gen

        # Backpropagate the total loss
        total_loss.backward()

    def optimize_parameters(self):
        """
            Update network weights by performing a forward pass, then optimizing the discriminator and generator.
        """
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # Optimize the generator
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def save_results(self, save_data, results_dir="./eval_results", data_name='none', data_ext='jpg'):
        """
            Save the training or testing results to disk.
        """

        for i, img_path in enumerate(self.image_paths):
            print(f'Processing image: {img_path}')
            # Extract the base name without the directory path and extension
            base_name = os.path.splitext(ntpath.basename(img_path))[0]
            img_name = f"{base_name}_{data_name}.{data_ext}"

            # Construct the full path for saving the image
            full_img_path = os.path.join(results_dir, img_name)

            # Convert the tensor to a numpy image and save
            img_numpy = tensor2im(save_data[i])
            save_image(img_numpy, full_img_path)

    def save_networks(self, epoch):
        """
            Save all the networks to the disk.
        """
        save_filename = f"{epoch}_generator.pth"
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(self.generator.cpu().state_dict(), save_path)
        self.generator.to(self.device)

        save_filename = f"{epoch}_generator.pth"
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(self.discriminator.cpu().state_dict(), save_path)
        self.discriminator.to(self.device)

    def load_networks(self, epoch):
        """
            Load all the networks from the disk.
        """
        for name, net in zip(["generator", "discriminator"], [self.generator, self.discriminator]):
            filename = f"{epoch}_{name}.pth"
            path = os.path.join(self.save_dir, filename)

            if not os.path.isfile(path):
                warnings.warn(f"Checkpoint not found for network {name} at {path}", RuntimeWarning)
                continue

            state_dict = torch.load(path, map_location=self.device)
            model_dict = net.state_dict()

            # Filter out unnecessary keys
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            # Update current model state dict
            model_dict.update(state_dict)
            net.load_state_dict(model_dict)

            print(f"Loaded {name} from {filename}")

            if not self.is_train:
                net.eval()


# From resource code -----------------------------------------------------------------------

def tensor2im(image_tensor, bytes=255.0, need_dec=False, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()

    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if need_dec:
        image_numpy = decode_labels(image_numpy.astype(int))
    else:
        image_numpy = (image_numpy + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)

# label color
label_colours = [(0, 0, 0)
    , (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0), (0, 0, 85), (0, 119, 221), (85, 85, 0),
                 (0, 85, 85), (85, 51, 0), (52, 86, 128), (0, 128, 0)
    , (0, 0, 255), (51, 170, 221), (0, 255, 255), (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)]

def decode_labels(mask, num_images=1, num_classes=20):
    """
        Decode batch of segmentation masks.

        Args:
          mask: result of inference after taking argmax.
          num_images: number of images to decode from the batch.
          num_classes: number of classes to predict (including background).

        Returns:
          A batch with num_images RGB images of the same size as the input.
    """
    h, w, c = mask.shape
    # assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((h, w, 3), dtype=np.uint8)

    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    tmp = []
    tmp1 = []
    for j_, j in enumerate(mask[:, :, 0]):
        for k_, k in enumerate(j):
            # tmp1.append(k)
            # tmp.append(k)
            if k < num_classes:
                pixels[k_, j_] = label_colours[k]
    # np.save('tmp1.npy', tmp1)
    # np.save('tmp.npy',tmp)
    outputs = np.array(img)
    # print(outputs[144,:,0])
    return outputs

def save_image(image_numpy, image_path):
    # Handle grayscale images (with a single channel)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.squeeze(axis=2)  # Remove the channel dimension

    # Handle images with 8 channels by converting them to a label map
    elif image_numpy.shape[2] == 8:
        image_numpy = np.argmax(image_numpy, axis=2)  # Convert channel dimension to label map
        image_numpy = decode_labels(image_numpy)  # Assume decode_labels returns an RGB image

    # Save the image
    image = Image.fromarray(image_numpy)
    image.save(image_path)

# From resource code -----------------------------------------------------------------------