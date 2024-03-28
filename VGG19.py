import torch
import torch.nn as nn
from torchvision import models


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice_indices = [0, 2, 4, 7, 9, 12, 14, 16, 18, 21, 23, 25, 27, 30, 32, 34]

        self.features = []
        for idx in range(len(self.slice_indices) - 1):
            module = nn.Sequential()
            for layer_idx in range(self.slice_indices[idx], self.slice_indices[idx + 1]):
                module.add_module(str(layer_idx), vgg_pretrained_features[layer_idx])
            self.add_module(f'relu{idx // 4 + 1}_{idx % 4 + 1}', module)
            self.features.append(module)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        results = {}
        for idx, model in enumerate(self.features):
            x = model(x)
            layer_name = f'relu{idx // 4 + 1}_{idx % 4 + 1}'
            results[layer_name] = x
        return results
