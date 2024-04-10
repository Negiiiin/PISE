import torch
import lpips


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

import numpy as np
from scipy.linalg import sqrtm

def calc_fid(feature_vectors_real, feature_vectors_fake):
    # # Convert PyTorch tensors to numpy and reshape to 2D arrays if necessary
    # real_features_np = real_features.detach().cpu().numpy()
    # gen_features_np = gen_features.detach().cpu().numpy()
    #
    # # If features have more than 2 dimensions, reshape them (flattening feature vectors)
    # if real_features_np.ndim > 2:
    #     real_features_np = real_features_np.reshape(real_features_np.shape[0], -1)
    # if gen_features_np.ndim > 2:
    #     gen_features_np = gen_features_np.reshape(gen_features_np.shape[0], -1)
    #
    # # Calculate the mean and covariance of the real and generated features
    # mu1, sigma1 = real_features_np.mean(axis=0), np.cov(real_features_np, rowvar=False)
    # mu2, sigma2 = gen_features_np.mean(axis=0), np.cov(gen_features_np, rowvar=False)
    #
    #
    # ssdiff = np.sum((mu1 - mu2) ** 2.0)
    #
    # covmean = sqrtm(sigma1.dot(sigma2))
    #
    # if np.iscomplexobj(covmean):
    #     covmean = covmean.real
    #
    # fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    #
    # return fid
    return 0
    # Move tensors to CPU and convert to NumPy if they're not already
    if feature_vectors_real.is_cuda or str(feature_vectors_real.device).startswith('mps'):
        feature_vectors_real = feature_vectors_real.cpu().numpy()
    else:
        feature_vectors_real = feature_vectors_real.numpy()

    if feature_vectors_fake.is_cuda or str(feature_vectors_fake.device).startswith('mps'):
        feature_vectors_fake = feature_vectors_fake.cpu().numpy()
    else:
        feature_vectors_fake = feature_vectors_fake.numpy()

    # Reshape the feature vectors to 2D if they're not already
    feature_vectors_real = feature_vectors_real.reshape(feature_vectors_real.shape[0], -1)
    feature_vectors_fake = feature_vectors_fake.reshape(feature_vectors_fake.shape[0], -1)


    # Calculate mean and covariance of feature vectors
    mu1, sigma1 = feature_vectors_real.mean(axis=0), np.cov(feature_vectors_real, rowvar=False)
    mu2, sigma2 = feature_vectors_fake.mean(axis=0), np.cov(feature_vectors_fake, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid
