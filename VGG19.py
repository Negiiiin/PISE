import torch
import torch.nn as nn
from torchvision import models

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.features = torch.nn.ModuleList()
        self.feature_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]

        for i in range(len(self.feature_indices) - 1):
            start, end = self.feature_indices[i], self.feature_indices[i + 1]
            self.features.append(torch.nn.Sequential(*[vgg_pretrained_features[j] for j in range(start, end)]))

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for feature_layer in self.features:
            x = feature_layer(x)
            features.append(x)

        feature_names = ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
                         'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4', 'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4']
        out = {name: feature for name, feature in zip(feature_names, features)}
        return out