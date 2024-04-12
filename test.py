import time

import numpy
from mpmath import sqrtm

import data as Dataset
import torch
from util import visualizer
from options.test_options import TestOptions
from torchvision import models, transforms, datasets
from final_model import *
import basic_blocks
from eval_metric import calculate_fid, get_fid_features
import tensorflow as tf
from torchvision.models.inception import inception_v3


# Setup TensorBoard writer
summary_writer = tf.summary.create_file_writer('./logsTensorBoardTest/')


# Function to attach hooks for logging gradients
def attach_gradient_logging_hooks(model):
    for name, module in model.named_modules():
        # Skip if it's the whole model itself and not a layer
        if name == "":
            continue

        def hook_fn(module, grad_input, grad_output, prefix=name):
            # Log gradient information
            print(f"--- Gradients for layer: {prefix} ---")
            for i, g in enumerate(grad_input):
                if g is not None:
                    print(f"Grad Input {i}: Mean = {g.mean()}, Std = {g.std()}")
            for i, g in enumerate(grad_output):
                if g is not None:
                    print(f"Grad Output {i}: Mean = {g.mean()}, Std = {g.std()}")
            print("---------------------------------")

        # Register the hook
        module.register_full_backward_hook(hook_fn)


def tensor2im(image_tensor, bytes=255.0, need_dec=False, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().detach().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().detach().numpy()

    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if need_dec:
        image_numpy = decode_labels(image_numpy.astype(int))
    else:
        image_numpy = (image_numpy + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)

def preprocess_for_inception(tensor):
    """
    Preprocess the tensor from [-1, 1] to the format expected by Inception V3.
    """
    # Scale from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2

    # Normalize using ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps
    covmean = sqrtm(sigma1.dot(sigma2), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return (diff.dot(diff)) + np.trace(sigma1 + sigma2 - 2.0 * covmean)

def calculate_activation_statistics(images_tensor, model, device):
    model.eval()
    with torch.no_grad():
        pred = model(images_tensor.to(device))
    pred = pred.detach().cpu().numpy()
    mu = np.mean(pred, axis=0)
    sigma = np.cov(pred, rowvar=False)
    return mu, sigma
def calculate_fid_with_tensor(model, real_images_tensor, fake_images_tensor, eps=1e-6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Preprocess the tensors for the Inception model
    real_images_tensor = preprocess_for_inception(real_images_tensor)
    fake_images_tensor = preprocess_for_inception(fake_images_tensor)

    mu_real, sigma_real = calculate_activation_statistics(real_images_tensor, model, device)
    mu_fake, sigma_fake = calculate_activation_statistics(fake_images_tensor, model, device)

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid


if __name__ == '__main__':

    opt = TestOptions().parse()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        use_multi_gpu = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        use_multi_gpu = False
    else:
        device = torch.device('cpu')
        use_multi_gpu = False

    # opt.isTrain = False
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('training images = %d' % dataset_size)



    keep_training = False

    visualizer = visualizer.Visualizer(opt)
    opt.device = device
    model = Final_Model(opt)


    model.load_networks(opt.gen_path, opt.dis_path)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    sum_result_psnr, sum_lpips_result, sum_fid_result = 0, 0, 0


    #FID setup

    model_v3 = models.inception_v3(pretrained=True, transform_input = False,init_weights=False).to(device)
    model_v3.fc = torch.nn.Identity()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_ground_truth = np.zeros((dataset_size, 2048))
    features_generated = np.zeros((dataset_size, 2048))
    batch_size = opt.batchSize
    with torch.no_grad():
        model_v3.eval()
        for i, data in enumerate(dataset):
            model.set_input(data)
            print(f'testing {i} / {len(dataset)}')
            generated_image, result_psnr, lpips_result = model.test_phase(i)

            sum_result_psnr += result_psnr * model.input_P1.shape[0]
            sum_lpips_result += lpips_result * model.input_P1.shape[0]

            # input_tensor = (model.input_P2 + 1) / 2
            # generated_image = (generated_image + 1) / 2
            pred1 = model_v3(model.input_P2)
            features_ground_truth[i * batch_size: i * batch_size + model.input_P2.size(0)] = pred1.cpu().numpy()
            pred = model_v3(generated_image)
            features_generated[i * batch_size: i * batch_size + generated_image.size(0)] = pred.cpu().numpy()

            fid_value = calculate_fid(pred1, pred)
            print(f'FID score: {fid_value}')

        print('average psnr: %.4f' % (sum_result_psnr / dataset_size))
        print('average lpips: %.4f' % (sum_lpips_result / dataset_size))

        print('Calculating FID score')
        print(features_ground_truth.shape, features_generated.shape)
        fid_value = calculate_fid(features_ground_truth, features_generated)
        print(f'FID score: {fid_value}')

        fid_value = calculate_fid_with_tensor(model_v3, features_ground_truth, features_generated)
        print(f'FID score: {fid_value}')
    #     res
    # testing 8569 / 8570
    # average psnr: 10.7624
    # average lpips: 0.2490
    # average fid: 0.0000

