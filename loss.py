import torch
import torch.nn as nn
from VGG19 import VGG19
import torch.nn.functional as F
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

# TODO
class VGG_Loss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(VGG_Loss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return content_loss, style_loss

class Cross_Entropy_Loss2d(nn.Module):
    # def __init__(self, weight=None, ignore_index=255, reduction='mean'):
    #     super(Cross_Entropy_Loss2d, self).__init__()
    #     self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    #
    # def forward(self, inputs, targets):
    #     return self.loss(inputs, targets)
    def __init__(self, weight=None, ignore_index=255):
        super(Cross_Entropy_Loss2d, self).__init__()
        self.ignore_index = ignore_index
        self.loss_function = nn.NLLLoss(weight=weight, reduction='mean', ignore_index=ignore_index)

    def forward(self, inputs, targets):
        # Applying LogSoftmax to the inputs
        inputs = F.log_softmax(inputs, dim=1)
        # Calculating NLL loss
        loss = self.loss_function(inputs, targets)
        return loss



