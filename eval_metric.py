import os

import torch
import lpips
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from scipy import linalg
from PIL import Image
import glob
import re


def psnr(target, prediction, max_pixel=1.0):
    """
    Computes the Peak Signal-to-Noise Ratio between two images.

    Parameters:
        target (torch.Tensor): The ground truth tensor.
        prediction (torch.Tensor): The tensor produced by the model.
        max_pixel (float): The maximum pixel value in the images. Default is 1.0 for normalized images.

    Returns:
        float: The PSNR value.
    """
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        # Avoid division by zero
        return torch.tensor(float('inf'))
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


def calc_lpips(original, generated, opt):
    lpips_model = lpips.LPIPS(net='vgg').to(opt.device)

    similarity = lpips_model(original, generated)
    return similarity[0,0,0]


#FID processing
def numerical_sort(filename):
    """
    Extracts the numerical part at the beginning of the filename and returns it as an integer.
    Assumes filenames are in the format 'num_restOfString.ext'.
    """
    match = re.search(r'Test:(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 0  # Default value if the pattern doesn't match


def get_fid_features(batch_size=1, dims=2048, device='mps', dataloader=None, dataset_size=8570, directory=None):
    model = models.inception_v3(pretrained=True).to(device)
    model.fc = torch.nn.Identity()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if dataloader is not None:
        model.eval()
        features = np.zeros((dataset_size, dims))
        for i, (images) in enumerate(dataloader):
            image = images['P2'].to(device)
            path = images['P2_path']
            print(f'Processing image {path}')
            with torch.no_grad():
                pred = model(image)
                features[i * batch_size: i * batch_size + image.size(0)] = pred.cpu().numpy()
    else:
        model.eval()
        features = []
        filenames = sorted(glob.glob(os.path.join(directory, '*.jpg')),
                           key=lambda x: numerical_sort(os.path.basename(x)))

        for filename in filenames:  # assuming images are in JPG format
            print(f'Processing image {filename}')
            image = Image.open(filename).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(image)
                features.append(pred.cpu().numpy())

        features = np.vstack(features)

    return features

def calculate_fid(real_features, generated_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


