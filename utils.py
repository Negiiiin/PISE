import torch.nn as nn
import torch
from torch.nn.utils.spectral_norm import spectral_norm
import torchvision
import os


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
    def __init__(self, conv_in_channel, conv_out_channel, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.LeakyReLU(), use_spect=False, use_coord=False):
        super(Encoder_Block, self).__init__()

        # Convolutional layer parameters From code
        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        ############################################

        # Initialize convolutional layers
        self.conv1 = self._coord_conv(conv_in_channel, conv_out_channel, use_spect, use_coord, **kwargs_down)
        self.conv2 = self._coord_conv(conv_out_channel, conv_out_channel, use_spect, use_coord, **kwargs_fine)

        # Sequential model
        self.model = nn.Sequential(norm_layer(conv_in_channel),
                                   activation_layer,
                                   self.conv1,
                                   norm_layer(conv_out_channel),
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

class Basic_Encoder(nn.Module):
    """
        A basic encoder module to extract feature representations. The module consists
        of sequential encoder blocks that downsample the input feature map and increase
        its depth.
    """

    def __init__(self, input_nc, generator_filter_num=64, use_spect=True):
        super(Basic_Encoder, self).__init__()

        self.encoder_block_1 = Encoder_Block(input_nc, generator_filter_num*2, use_spect=use_spect)
        self.encoder_block_2 = Encoder_Block(generator_filter_num*2, generator_filter_num*4, use_spect=use_spect)

    def forward(self, x):
        """
            Defines the forward pass of the encoder.

            :param input: Input tensor to the encoder.
            :return: Output tensor after passing through the encoder blocks.
        """
        x = self.encoder_block_1(x)
        x = self.encoder_block_2(x)
        return x

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
                 activation=nn.LeakyReLU(), learnable_shortcut=False, use_spect=False):
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
        if self.learnable_shortcut:
            self.shortcut = nn.Sequential(
                Coord_Conv(input_nc, output_nc, use_spect=use_spect, **conv_shortcut_params)
            )
        else:
            self.shortcut = nn.Identity()

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

        if use_spect:
            spec_norm_func = spectral_norm
        else:
            spec_norm_func = nn.Identity

        # Main convolutional layers with optional spectral normalization
        layers = [
            norm_layer(input_nc),
            spec_norm_func(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1)),
            activation,
            norm_layer(hidden_nc),
            spec_norm_func(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1))
        ]

        self.model = nn.Sequential(*layers)

        # Shortcut connection with optional spectral normalization
        self.shortcut = nn.Sequential(
            spec_norm_func(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1))
        )

    def forward(self, x):
        return self.model(x) + self.shortcut(x)

class Hard_Encoder(nn.Module):
    """
        Hard Encoder with configurable normalization, activation, and spectral normalization.
        Uses EncoderBlocks and ResBlockDecoders for encoding, and Gated Convolutions for feature modulation.
    """
    def __init__(self, input_nc, generator_filter_num=64, norm_layer=nn.InstanceNorm2d(affine=True),
                 activation=nn.LeakyReLU(0.1), use_spect=True):
        super(HardEncoder, self).__init__()

        # Define encoder blocks
        self.encoder_blocks = nn.Sequential(
            Encoder_Block(input_nc, generator_filter_num*2, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*2, generator_filter_num*4, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*4, generator_filter_num*4, norm_layer, activation, use_spect),
            Encoder_Block(generator_filter_num*4, generator_filter_num*4, norm_layer, activation, use_spect)
        )

        # Define residual blocks in the decoder
        self.res_blocks = nn.Sequential(
            Res_Block_Decoder(generator_filter_num*4, generator_filter_num*4, generator_filter_num*4, norm_layer, activation, use_spect),
            Res_Block_Decoder(generator_filter_num*4, generator_filter_num*4, generator_filter_num*4, norm_layer, activation, use_spect)
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