import time
import data as Dataset
import torch
from util import visualizer
from options.train_options import TrainOptions

from final_model import *
import basic_blocks

import tensorflow as tf

# Setup TensorBoard writer
summary_writer = tf.summary.create_file_writer('./logsTensorBoard/')


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
    max_epochs = 100

    opt = TrainOptions().parse()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        use_multi_gpu = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        use_multi_gpu = False
    else:
        device = torch.device('cpu')
        use_multi_gpu = False
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('training images = %d' % dataset_size)

    keep_training = True

    epoch = 0


    visualizer = visualizer.Visualizer(opt)
    opt.device = device
    opt.isTrain = True
    model = Final_Model(opt)

    if opt.continue_train:
        model.load_networks(opt.gen_path, opt.dis_path)
        part_after_slash = opt.gen_path.split("/")[-1]

        # Splitting the isolated part by underscores to extract Num1 and Num2
        parts = part_after_slash.split("_")
        # Extracting num1 and num2 assuming they are integers
        epoch = int(parts[0]) - 1

    # Attach hooks to all layers
    # attach_gradient_logging_hooks(model)

    # model.clear_or_create_directory("logs")
    # model.clear_or_create_directory("fashion_data/eval_results")

    # training process
    while (epoch < max_epochs):
        epoch_start_time = time.time()
        epoch += 1
        print('\n Training epoch: %d' % epoch)


        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            model.set_input(data)
            model.optimize_parameters()

            if i % opt.generate_img == 0:
                for param in model.generator.parameters():
                    param.requires_grad = False
                for param in model.discriminator.parameters():
                    param.requires_grad = False
                print('generating images of iterations %d at epoch %d' % (i, epoch))
                model.test2(i, epoch)

                eval_results = model.get_loss_results()
                visualizer.print_current_eval(epoch, i, eval_results)
                visualizer.tensorboard_log(epoch, i, eval_results, summary_writer)
                visualizer.tensorboard_weights_and_grads(model.generator, epoch, i, summary_writer, "generator")
                visualizer.tensorboard_weights_and_grads(model.discriminator, epoch, i, summary_writer, "discriminator")
                if opt.display_id > 0:
                    visualizer.plot_current_score(i, eval_results)

                for param in model.generator.parameters():
                    param.requires_grad = True
                for param in model.discriminator.parameters():
                    param.requires_grad = True

            if i % opt.save_net == 0:
                model.save_networks(epoch, i)

    print('\nEnd Training')
