###############################################################################
# This file contains the base model class which will be inherited by 
# other child model classes
###############################################################################

import os
import torch
from torch.optim import lr_scheduler
import util.util as util
from collections import OrderedDict

class BaseModel():
    def name(self):
        return 'BaseModel'
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        
        # When model doesn't vary, we set torch.backends.cudnn.benchmark to get the benefit 
        if opt.resize_or_crop != 'scale_width': 
            torch.backends.cudnn.benchmark = True
            
        self.loss_names = []
        self.model_names = []
        self.image_paths = []
   
    def convert_tensor_type(self, data):     
        cuda_tensor = torch.cuda.FloatTensor if len(self.opt.gpu_ids) > 0 else torch.FloatTensor
        return data.type(cuda_tensor)
            
    def set_input(self, input):
        self.input_img = self.convert_tensor_type(input['shadowfull'])
        self.shadow_mask = self.convert_tensor_type(torch.round((input['shadowmask']+1.0)/2)*2-1)
        self.shadowfree_img = self.convert_tensor_type(input['shadowfree'])
        self.skin_mask = self.convert_tensor_type(torch.round((input['skinmask']+1.0)/2)*2-1) if self.opt.use_skinmask else None 
    
    def forward(self):
        pass
    
    def get_prediction(self, input_img, skin_mask = None):
        self.input_img = self.convert_tensor_type(input_img)
        self.skin_mask = self.convert_tensor_type(skin_mask) if skin_mask != None else skin_mask
        self.forward()

        self.result = dict()
        self.result['final']= self.fake_free_shadow_image
        self.result['phase1'] = self.fake_shadow_image 

    # Load and print networks; create schedulers
    def setup(self, opt, parser=None):
        print(self.name)
        if self.isTrain:
            print(self.optimizers)
            self.schedulers = [self.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            
        if (not self.isTrain) or (opt.continue_train):
            print("LOADING %s"%(self.name))
            self.load_networks(opt.epoch)
        self.print_networks(opt)
        
    def get_scheduler(self, optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'shadow_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70000,90000,13200], gamma=0.3)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler

    def optimize_parameters(self):
        pass

    # Save and load the networks
    def save_networks(self, epoch):
        for model_name in self.model_names:    
            save_filename = '%s_net_%s.pth' % (epoch, model_name)
            save_path = os.path.join(self.save_dir, save_filename)

            net = getattr(self, 'net' + model_name)
            if len(self.gpu_ids) > 0 and torch.cuda.is_available(): # The case for multiple GPUs
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0]) 
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]

        if i + 1 == len(keys):  # At the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module,key), keys, i + 1) 

    def load_networks(self, epoch):
        for name in self.model_names:
            load_filename = '%s_net_%s.pth' % (epoch, name)
            load_path = os.path.join(self.save_dir, load_filename)

            # The case for multiple GPUs
            net = getattr(self, 'net' + name)
            
            print('loading the model from %s' % load_path)
            device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if len(self.opt.gpu_ids)>0 else torch.device('cpu')
            state_dict = torch.load(load_path, map_location=str(device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
                
            # Loop all keys to find and remove checkpoints of InstanceNorm
            for key in list(state_dict.keys()):
                if 'module' in key  and len(self.gpu_ids) == 0:
                    state_dict[key.replace('.module', '')] = state_dict[key]
                    del state_dict[key]
                    key = key.replace('.module', '')
                self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            net.load_state_dict(state_dict)
                
    # Print network information
    def print_networks(self, opt):
        message = ""
        message += '---------- Networks initialized -------------\n'
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                message += "{}\n".format(net)
                message += "[Network {}] Total number of parameters : {:.3f} M\n".format(name, num_params / 1e6)
        message += '-------GPU ids: {} -----------\n'.format(self.gpu_ids)
        message += '-----------------------------------------------'
        print(message)

        # Save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'network.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            
    def update_learning_rate(self,loss=None):
        for scheduler in self.schedulers:
            if not loss:
                scheduler.step()
            else:
                scheduler.step(loss)
        lr = self.optimizers[0].param_groups[0]['lr']
        return lr
        
    # Set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                     
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if hasattr(self,'loss_'+name):
                errors_ret[name] = float("%.4f" % getattr(self, 'loss_' + name))        
        return errors_ret