import torch
import torch.nn as nn
from VGG19 import VGG19

class Adversarial_Loss(nn.Module):
    """
        Least Squares GAN Adversarial loss
        Reference: https://arxiv.org/abs/1611.04076
    """
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(Adversarial_Loss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.criterion = nn.MSELoss()

    def __call__(self, outputs, is_real):
        labels = self.real_label if is_real else self.fake_label
        labels = labels.expand_as(outputs)
        return self.criterion(outputs, labels)


class VGG_Loss(nn.Module):
    """
        Perceptual loss based on VGG19 features for content and style representation.
        References:
        - https://arxiv.org/abs/1603.08155
        - https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    def __init__(self, weights=None):
        super(VGG_Loss, self).__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        G = torch.bmm(f, f.transpose(1, 2)) / (h * w * ch)
        return G

    def forward(self, x, y):
        # Extract features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute content loss
        content_loss = sum(self.weights[i] * self.criterion(x_vgg[f'relu{i+1}_1'], y_vgg[f'relu{i+1}_1']) for i in range(5))

        # Compute style loss
        style_layers = ['relu2_2', 'relu3_4', 'relu4_4', 'relu5_2']
        style_loss = sum(self.criterion(self.compute_gram(x_vgg[layer]), self.compute_gram(y_vgg[layer])) for layer in style_layers)

        return content_loss, style_loss


class Cross_Entropy_Loss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(Cross_Entropy_Loss2d, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)



