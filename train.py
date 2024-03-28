import time
import data as Dataset
import torch
from . import visualizer


import utils

opt = {
    "batchSize": 16,
    "serial_batches": False,
    "nThreads": 8,
    "isTrain": True,
    "max_epochs": 40,
    "gpu_ids": 0,
    "display_id": 0,
    "display_winsize": 256,
    "name": "experiment_name",
    "no_html": True,
    # "display_env":,
    "display_single_pane_ncols": 0,
    "checkpoints_dir": "./results",
    "save_iters_freq": 5000
}

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')  # Use MPS on supported Macs
else:
    device = torch.device('cpu')

dataset = Dataset.create_dataloader(opt)

dataset_size = len(dataset) * opt["batchSize"]
print('training images = %d' % dataset_size)

keep_training = True

epoch = 0

visualizer = visualizer.Visualizer(opt)
model = utils.Final_Model(gpu_ids=opt["gpu_ids"], device=device, save_dir=opt["checkpoints_dir"])

# training process
while (epoch < opt["max_epochs"]):
    epoch_start_time = time.time()
    epoch += 1
    print('\n Training epoch: %d' % epoch)


    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        model.set_input(data)
        model.optimize_parameters()

        # save the model every <save_iter_freq> iterations to the disk
        if i % opt["save_iters_freq"] == 0:
            print('saving the model of iterations %d at epoch %d' % (i, epoch))
            model.save_networks(epoch*len(dataset)+i)

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

        if total_iteration % opt.eval_iters_freq == 0:
            model.eval()
            if hasattr(model, 'eval_metric_name'):
                eval_results = model.get_current_eval_results()
                visualizer.print_current_eval(epoch, total_iteration, eval_results)
                if opt.display_id > 0:
                    visualizer.plot_current_score(total_iteration, eval_results)

        # save the latest model every <save_latest_freq> iterations to the disk
        if total_iteration % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
            model.save_networks('latest')
"""

    # model.update_learning_rate() # TODO

print('\nEnd Training')
