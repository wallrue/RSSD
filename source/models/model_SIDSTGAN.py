###############################################################################
# This file contains the model class for the combination of STGAN and SID
# STGAN is in charge of detecting shadow. SID is in charge of removing shadow.
###############################################################################

import torch 
from .base_model import BaseModel
from .network import network_GAN
from .network import network_STGAN
from .network.network_SID import define_SID

class SIDSTGANModel(BaseModel):
    def name(self):
        return 'Shadow Removal via Shadow Image Decomposition & STGAN'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch')
        parser.set_defaults(input_nc=3, output_nc=3)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['G1_GAN', 'G1_L1', 'D1_real', 'D1_fake', 
                           'G2_param', 'G2_L1']
        self.model_names = ['G1', 'G2']
        self.netG1 = network_STGAN.define_STGAN(opt, 3 + self.opt.use_skinmask, 1, net_g = opt.netG, net_d = opt.netD)
        self.netG2 = define_SID(opt, net_g = opt.netS, net_m = opt.netG)
        
        if self.isTrain:
            # Define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.GAN_loss = network_GAN.GANLoss(opt.gpu_ids)
        
            # Initialize optimizers
            self.optimizer_G1 = torch.optim.Adam(self.netG1.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_G2 = torch.optim.Adam([{'params': self.netG2.netG.parameters()},
                                                  {'params': self.netG2.netM.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_D = torch.optim.Adam(self.netG1.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G1, self.optimizer_G2, self.optimizer_D]
   
    def set_input(self, input):
        BaseModel.set_input(self, input)
        self.shadow_param = self.convert_tensor_type(input['shadowparams'])
        
    def forward(self):
        # Compute output of generator 1
        self.inputNet1 = torch.cat((self.input_img, self.skin_mask), 1) if self.opt.use_skinmask else self.input_img
        self.fake_shadow_image = self.netG1.forward_G(self.inputNet1)
        
        # Compute output of generator 2
        self.compounded_shadow = torch.cat((self.fake_shadow_image, self.skin_mask), 1) if self.opt.use_skinmask else self.fake_shadow_image 
        self.shadow_param_pred, self.alpha_pred, self.fake_free_shadow_image = self.netG2(self.input_img, self.compounded_shadow)
                
    def forward_D(self):
        fake_AB = torch.cat((self.inputNet1, self.fake_shadow_image), 1)
        real_AB = torch.cat((self.inputNet1, self.shadow_mask), 1)                                                             
        self.pred_fake, self.pred_real = self.netG1.forward_D(fake_AB.detach(), real_AB)
        
    def backward_D(self):      
        self.loss_D1_fake = self.GAN_loss(self.pred_fake, target_is_real = 0) 
        self.loss_D1_real = self.GAN_loss(self.pred_real, target_is_real = 1)
        
        lambda_ = 0.5;
        loss_D1 = self.loss_D1_fake + self.loss_D1_real
        self.loss_D = lambda_ * loss_D1
        self.loss_D.backward()

    def backward_G1(self):
        self.loss_G1_GAN = self.GAN_loss(self.pred_fake, target_is_real = 1)                                           
        self.loss_G1_L1 = self.criterionL1(self.fake_shadow_image, self.shadow_mask)
        self.loss_G1 = self.loss_G1_L1*0.1 + self.loss_G1_GAN 
        self.loss_G1.backward(retain_graph=False)
 
    def backward_G2(self):
        lambda_ = 2
        self.shadow_param[:,[1,3,5]] = (self.shadow_param[:,[1,3,5]])/2 - 1.5
        self.loss_G2_param = self.criterionL1 (self.shadow_param_pred, self.shadow_param) * lambda_ 
        self.loss_G2_L1 = self.criterionL1 (self.fake_free_shadow_image, self.shadowfree_img) * lambda_
        self.loss_G2 = self.loss_G2_param + self.loss_G2_L1
        self.loss_G2.backward(retain_graph=True)
    
    def get_prediction(self, input_img, skin_mask = None):
        BaseModel.get_prediction(self, input_img, skin_mask)
        return self.result
    
    def optimize_parameters(self):
        self.forward()
        
        self.set_requires_grad(self.netG1.netD, True)  # Enable backprop for D1, D2
        self.forward_D()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
        self.set_requires_grad(self.netG1.netD, False) # Freeze D
        self.forward_D()
                
        self.optimizer_G2.zero_grad()
        self.backward_G2()
        self.optimizer_G2.step()
      
        self.optimizer_G1.zero_grad()
        self.backward_G1()
        self.optimizer_G1.step()


