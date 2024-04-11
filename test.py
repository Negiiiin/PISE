import time
import data as Dataset
import torch
from util import visualizer
from options.test_options import TestOptions

from final_model import *
import basic_blocks
from eval_metric import calculate_fid, get_fid_features
import tensorflow as tf

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



if __name__ == '__main__':
    max_epochs = 30

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


    # Calculate FID
    with torch.no_grad():


        generated_dir = './fashion_data/test_output'
        generated_features = get_fid_features(device=device, directory=generated_dir)
        real_features = get_fid_features(device=device, dataloader=dataset)

        print('Calculating FID score')
        print(real_features.shape, generated_features.shape)
        fid_value = calculate_fid(real_features, generated_features)
        print(f'FID score: {fid_value}')


    keep_training = False

    visualizer = visualizer.Visualizer(opt)
    opt.device = device
    model = Final_Model(opt)


    model.load_networks(opt.gen_path, opt.dis_path)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    sum_result_psnr, sum_lpips_result, sum_fid_result = 0, 0, 0
    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            print(model.input_P1.shape)
            result_psnr, lpips_result = model.test_phase(i)
            sum_result_psnr += result_psnr
            sum_lpips_result += lpips_result
            print('testing %d / %d' % (i, len(dataset)))

        print('average psnr: %.4f' % (sum_result_psnr / dataset_size))
        print('average lpips: %.4f' % (sum_lpips_result / dataset_size))
    #     res
    # testing 8569 / 8570
    # average psnr: 10.7624
    # average lpips: 0.2490
    # average fid: 0.0000

