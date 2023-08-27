###############################################################################
# This file is used for training models. 
# There are three major parameters in main function which needs being defined: 
# - dataset_dir: the root folder which contains dataset (with the datasetname)
# - checkpoints_dir: the folder to save checkpoints after training
# - training_dict: the model to be trained (with the datasetname)
# dataset is trained with models in training_dict in a run
###############################################################################

import sys
import os
import ast
import time
import numpy as np
import matplotlib.pyplot as plt
from options.train_options import TrainOptions
from data import CustomDatasetDataLoader
from models import create_model
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #Fix error on computer
    
def progressbar(it, info_dict, size=60, out=sys.stdout):
    count = len(it)
    def show(j, batch_size):
        n = batch_size*j if batch_size*j < count else count
        x = int(size*n/count) 
        taken_time = time.time() - info_dict["start time"]
        print("\r{} [{}{}] {}/{} | {:.3f} secs".format(info_dict["epoch"], "#"*x, "."*(size-x), n, count, taken_time), 
                end='', file=out, flush=True) # Flushing for progressing bar in Python 3.0 
        sys.stdout.flush() # Flushing for progressing bar in Python 2.0 
        
    show(0, 1)
    for i, item in enumerate(it):
        yield i, item
        if i == 0: # Initialize batch_size value
            batch_size = len(list(item.values())[0])
        show(i+1, batch_size)
    print("", flush=True, file=out) # Do thing after ending iteration
    
def print_current_losses(log_dir, epoch, lr, iters, losses, t_comp, t_data):
    message = '{\"epoch\": %d, \"iters\": %d, \"lr\": %.6f, \"computing time\": %.3f, \"data_load_time\": %.3f ' % (epoch, iters, lr, t_comp, t_data)
    for k, v in losses.items():
        message += ', \"%s\": %.3f ' % (k, v)
    message += '}'

    print(" - " + log_dir[-9:] + " : " + message)  # print the message
    with open(log_dir, "a+") as log_file:
        log_file.write('%s\n' % message)  # save the message
        
def get_loss_file(log_dir): 
    with open(log_dir, "r+") as f:
        data = f.readlines()
    loss = {'train_loss': list()}
    for i in data:
        my_dict = ast.literal_eval(str(i))
        loss['train_loss'].append(my_dict['train_reconstruction'])
    return loss

def loss_figure(loss_folder):
    history = get_loss_file(os.path.join(loss_folder, 'train.log'))
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (20,10))
    N = 1
    ax.plot(running_mean(history['train_loss'], N), 'r-', label='training loss')
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(loss_folder, 'loss_figure.png'), dpi=300)
    
def train_loop(opt, model):
    for epoch in range(opt.epoch_count, 1 + opt.epoch_pause):
        # Dataset loading
        data_loader = CustomDatasetDataLoader(opt)
        dataset = data_loader.load_data()
        
        epoch_start_time = time.time()
        epoch_iter = 0
        t_comp, t_data = 0, 0
        train_losses = dict()

        dataset.working_subset = "main"
        progressbar_info = {"epoch": "epoch {}/{} ".format(epoch, opt.niter + opt.niter_decay), 
                            "start time": epoch_start_time}
        for i, data in progressbar(dataset, progressbar_info):
            iter_start_time = time.time() # Finishing loading data
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            t_comp += time.time() - iter_start_time # Computing time is from finishing loading data to now

            current_losses = model.get_current_losses()
            train_losses = {key: train_losses.get(key,0) + current_losses[key] for key in set(train_losses).union(current_losses)}
        n_train_losses = epoch_iter/opt.batch_size
        train_losses = {'train_reconstruction': train_losses["G2_L1"]/n_train_losses}
        current_lr = model.update_learning_rate()
        t_data = time.time() - epoch_start_time - t_comp
        print_current_losses(os.path.join(opt.checkpoints_dir, opt.name, 'train.log'), epoch, current_lr, \
                                 epoch_iter, train_losses, t_comp, t_data)
           
        # Saving model
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)
        
if __name__=='__main__':
    
    checkpoint_dir = os.path.join(os.getcwd(),"_checkpoints")    
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    train_options = TrainOptions()
    dataset_dir = os.path.join(os.path.join(os.getcwd(),"_database"), "data_creating")
    
    """ DEFINE EXPERIMENT """
    training_dict =[["SIDPAMIwISTGAN",  True],
                    ["SIDSTGAN",        True],
                    ["STGAN",           True],
                    ["SIDPAMIwISTGAN",  False],
                    ["SIDSTGAN",        False],
                    ["STGAN",           False],
                    ]

    """ RUN SECTION """
    for model_name, use_skinmask in training_dict:
        dataset_name = "rawsynthetic"
        print('============== Start training: dataset {}, model {} =============='.format(dataset_name, model_name))  
        train_options.dataset_mode = dataset_name
        train_options.data_root = dataset_dir
        train_options.checkpoints_root = checkpoint_dir      
        train_options.model_name = model_name
        opt = train_options.parse()
        
        opt.use_skinmask = use_skinmask
        if opt.use_skinmask:
            opt.name = opt.name + "_HandSeg"
        train_options.print_options(opt)

        # Model defination
        model = create_model(opt)
        model.setup(opt)
    
        # Training
        train_loop(opt, model)
        loss_figure(os.path.join(opt.checkpoints_dir, opt.name))