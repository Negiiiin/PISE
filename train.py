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
    max_epochs = 30

    opt = TrainOptions().parse()

    if torch.cuda.is_available():
        # Assuming you want to use GPU 0 and 1
        device_ids = [2, 3]
        # Specify the primary device for model initialization
        device = torch.device('cuda:0')
        # Note: DataParallel will replicate the model on multiple GPUs
        use_multi_gpu = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')  # Use MPS on supported Macs
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
    model = Final_Model(opt).to(opt.device)
    # if use_multi_gpu:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.init_weights()

    # Attach hooks to all layers
    # attach_gradient_logging_hooks(model)

    model.clear_or_create_directory("logs")
    model.clear_or_create_directory("fashion_data/eval_results")

    # training process
    while (epoch < max_epochs):
        epoch_start_time = time.time()
        epoch += 1
        print('\n Training epoch: %d' % epoch)


        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            model.set_input(data)
            # model.eval()
            # print('saving the model of iterations %d at epoch %d' % (i, epoch))
            # model.test(i, epoch)
            model.train()
            model.optimize_parameters()

            # if i % opt.save_iters_freq == 0:
            #     print('saving the model of iterations %d at epoch %d' % (i, epoch))
            #     iter_count = opt.which_iter + (epoch*len(dataset)+i)
            #     model.save_networks(iter_count)
            #     model.test(i, epoch)
            model.eval()
            print('saving the model of iterations %d at epoch %d' % (i, epoch))
            model.test(i, epoch)
            model.train()
            model.zero_grad()

            if i % opt.eval_iters_freq == 0:
                # model.eval()
                eval_results = model.get_loss_results()
                visualizer.print_current_eval(epoch, i, eval_results)
                visualizer.tensorboard_log(epoch, i, eval_results, summary_writer)
                visualizer.tensorboard_weights_and_grads(model.generator, epoch, i, summary_writer, "generator")
                visualizer.tensorboard_weights_and_grads(model.discriminator, epoch, i, summary_writer, "discriminator")
                if opt.display_id > 0:
                    visualizer.plot_current_score(i, eval_results)
                # model.train()

        """
            # display images on visdom and save images
            if total_iteration % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)
                if hasattr(model, 'distribution'):
                    visualizer.plot_current_distribution(model.get_current_dis())
    
            # print training loss and save logging information to the disk
            if total_iteration % opt.print_freq == 0:
                losses = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, total_iteration, losses, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(total_iteration, losses)
        """

        model.update_lr()

    print('\nEnd Training')
