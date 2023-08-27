###############################################################################
# This file contains Base Options (arguments, input params) should be defined 
# by user. This Option will be inherited by Train Options and Test Options.
###############################################################################

import argparse
import os
import torch
import models
from util import util

class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.model_name = ""        # Be modified in running file. Eg: "SID"
        self.checkpoints_root = ""
        self.dataset_mode = ""      # Be modified in running file. Eg: "shadowparam"
        self.data_root = ""         # Be modified in running file. Eg: ".../dataset/ABCD/train/"
        
        self.netG = 'unet_256'
        self.netS = 'RESNEXT'
        self.netD = 'n_layers'
        
    def initialize(self, parser):
        # Data loader argument
        parser.add_argument('--dataroot',  help='path to dataset')
        parser.add_argument('--dataset_mode', type=str, default='single', help='chooses kind of dataset loader. [single, shadowparam]')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')        
        parser.add_argument('--num_threads', type=int, default=2, help='# threads for loading data, num_workers')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        
        # Data transform argument
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=224, help='then crop to this size')
        
        # Model setup
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment as well as name of checkpoint sub-folder')
        parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='models are saved here')
        parser.add_argument('--input_nc', type=int, default=3, help='channels of input image')
        parser.add_argument('--output_nc', type=int, default=3, help='channels of output image')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        
        # Model training configuration
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.0002, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--save_epoch_freq', type=int, default=2, help='periods of saving checkpoints')

        self.initialized = True
        return parser
    
    def get_known(self, parser):
        # Data loader argument        
        parser.set_defaults(dataroot=self.data_root)
        parser.set_defaults(dataset_mode=self.dataset_mode) # This param will be modified in train.py

        # Model setup: in gather_options by models.get_option_setter()
        net_id_name = ""
        parser.set_defaults(name=self.model_name + "_" + self.dataset_mode + net_id_name)
        parser.set_defaults(model=self.model_name)
        parser.set_defaults(checkpoints_dir=os.path.join(self.checkpoints_root,"checkpoints_" + self.model_name))

        args, unknown = parser.parse_known_args()
        return args

    def gather_options(self):
        # Initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self.parser_pre = self.initialize(parser)

        # Get the basic options
        _ = self.get_known(self.parser_pre)
        
        # Modify model-related parser options
        model_option_setter = models.get_option_setter(self.model_name)
        parser = model_option_setter(self.parser_pre, self.isTrain)
        args = self.get_known(parser)
        
        self.parser = parser
        return args

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # Save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        #opt.model = self.model_name
        opt.isTrain = self.isTrain   # Train or test

        # Process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
 
        # Change gpu_ids from string_type to list_type
        if opt.gpu_ids == "all":
            opt.gpu_ids = list(range(torch.cuda.device_count()))
        else:
            str_ids = opt.gpu_ids.split(',')
            opt.gpu_ids = []
            for device_id in str_ids:
                device_id = int(device_id)
                if device_id in range(torch.cuda.device_count()):
                    opt.gpu_ids.append(device_id)
        opt.netG, opt.netS, opt.netD = self.netG, self.netS, self.netD
            
        self.opt = opt
        return self.opt
