import time
import data as Dataset
import torch
from util import visualizer
from options.train_options import TrainOptions

import utils

opt = TrainOptions().parse()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')  # Use MPS on supported Macs
else:
    device = torch.device('cpu')

dataset = Dataset.create_dataloader(opt)

dataset_size = len(dataset) * opt.batchSize
print('training images = %d' % dataset_size)

keep_training = True

epoch = 0

visualizer = visualizer.Visualizer(opt)
model = utils.Final_Model()
model.init_losses_and_optimizers()
model.init_weights()

# training process
while (epoch < opt.max_epochs):
    epoch_start_time = time.time()
    epoch += 1
    print('\n Training epoch: %d' % epoch)


    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        model.set_input(data)
        model.optimize_parameters()

        # save the model every <save_iter_freq> iterations to the disk
        if i % opt.save_iters_freq == 0:
            print('saving the model of iterations %d at epoch %d' % (i, epoch))
            model.save_networks(epoch*len(dataset)+i)

        if i % opt.eval_iters_freq == 0:
            model.eval()
            eval_results = model.get_loss_results()
            visualizer.print_current_eval(epoch, i, eval_results)
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

    # model.update_learning_rate() # TODO

print('\nEnd Training')
