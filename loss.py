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

class VGG_Loss(nn.Module):
    """
    Defines a VGG-based perceptual loss as described in:
    https://arxiv.org/abs/1603.08155
    Uses features from a VGG19 network to compute content and style losses between two images.
    """
    def __init__(self, vgg_model=VGG19(), weights=None):
        super(VGG_Loss, self).__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.vgg = vgg_model
        self.criterion = nn.L1Loss()
        self.weights = weights
        self.content_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.style_layers = ['relu2_2', 'relu3_4', 'relu4_4', 'relu5_2']

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        features = x.view(b, ch, h * w)
        G = torch.matmul(features, features.transpose(1, 2))
        return G.div_(h * w * ch)

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = sum(self.weights[i] * self.criterion(x_vgg[layer], y_vgg[layer])
                           for i, layer in enumerate(self.content_layers))

        style_loss = sum(self.criterion(self.compute_gram(x_vgg[layer]), self.compute_gram(y_vgg[layer]))
                         for layer in self.style_layers)

        return content_loss, style_loss

class Cross_Entropy_Loss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(Cross_Entropy_Loss2d, self).__init__()
        self.ignore_index = ignore_index
        self.loss_function = nn.NLLLoss(weight=weight, reduction='mean', ignore_index=ignore_index)

    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        loss = self.loss_function(inputs, targets)
        return loss



