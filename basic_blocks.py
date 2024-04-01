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
        self.model = nn.Sequential(norm_layer(conv_in_channel).float(),
                                   activation_layer,
                                   self.conv1.float(),
                                   norm_layer(hidden_channel).float(),
                                   activation_layer,
                                   self.conv2.float())

    def _coord_conv(self, in_channels, out_channels, use_spect, use_coord, **kwargs):
        """
            Helper function to create a CoordConv or Conv2d layer with optional spectral normalization.
        """
        conv_layer = Coord_Conv(in_channels, out_channels, use_spect=use_spect, **kwargs) if use_coord \
                     else nn.Conv2d(in_channels, out_channels, **kwargs)
        return conv_layer.float()

    def forward(self, x):
        x = x.float()
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
                 activation=nn.LeakyReLU(), learnable_shortcut=False, use_spect=False): # maybe we dont need shortcut option and always use coordConv
        super(Res_Block, self).__init__()

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
            self.shortcut_path = Coord_Conv(conv_in_channel=hidden_nc, conv_out_channel=output_nc, use_spect=use_spect, **conv_params)

    def forward(self, x):
        if self.shortcut:
            return self.model(x) + self.shortcut_path(x)
        else:
            return self.model(x)

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
            Res_Block(ngf*2, ngf*2, ngf*2, norm_layer, activation, False, use_spect),
            Res_Block_Decoder(ngf*2, ngf, ngf*2, norm_layer, activation, use_spect),
            Res_Block(ngf, ngf, ngf, norm_layer, activation, False, use_spect),
            norm_layer(ngf),
            activation,
            nn.ReflectionPad2d(int((kernel_size - 1) / 2)),
            Coord_Conv(ngf, output_nc, use_spect=use_spect, **kwargs),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)

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
        self.conv_gamma = nn.Conv2d(style_length, input_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv_beta = nn.Conv2d(style_length, input_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
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

class Res_Block_Encoder(nn.Module):
    """
    Residual Block for Encoder
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU(), use_spect=False):
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