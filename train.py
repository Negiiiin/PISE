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

if __name__ == '__main__':
    max_epochs = 30

    opt = TrainOptions().parse()

    if torch.cuda.is_available():
        # Assuming you want to use GPU 0 and 1
        device_ids = [2, 3]
        # Specify the primary device for model initialization
        device = torch.device('cuda:2')
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
    if use_multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.init_weights()

    # training process
    while (epoch < max_epochs):
        epoch_start_time = time.time()
        epoch += 1
        print('\n Training epoch: %d' % epoch)


        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            model.set_input(data)
            model.optimize_parameters()

            if i % opt.save_iters_freq == 0:
                print('saving the model of iterations %d at epoch %d' % (i, epoch))
                iter_count = opt.which_iter + (epoch*len(dataset)+i)
                model.save_networks(iter_count)
                model.test(i, epoch)

            if i % opt.eval_iters_freq == 0:
                model.eval()
                eval_results = model.get_loss_results()
                visualizer.print_current_eval(epoch, i, eval_results)
                visualizer.tensorboard_log(epoch, i, eval_results, summary_writer)
                if opt.display_id > 0:
                    visualizer.plot_current_score(i, eval_results)

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
