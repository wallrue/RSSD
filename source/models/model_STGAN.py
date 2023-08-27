###############################################################################
# This file contains the model class for STGAN
###############################################################################

import torch
from .base_model import BaseModel
from .network import network_GAN
from .network import network_STGAN

class STGANModel(BaseModel):
    def name(self):
        return 'Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch')
        parser.set_defaults(input_nc=3, output_nc=3)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['G1_GAN', 'G1_L1', 'G2_GAN', 'G2_L1',
                           'D1_real', 'D1_fake', 'D2_real', 'D2_fake']
        self.model_names = ['STGAN1', 'STGAN2']
        
        self.netSTGAN1 = network_STGAN.define_STGAN(opt, 3 + self.opt.use_skinmask, 1, net_g = opt.netG, net_d = opt.netD)
        self.netSTGAN2 = network_STGAN.define_STGAN(opt, 4 + self.opt.use_skinmask, 3, net_g = opt.netG, net_d = opt.netD)
        
        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.GAN_loss = network_GAN.GANLoss(opt.gpu_ids)
        
            # Initialize optimizers
            self.optimizer_G1 = torch.optim.Adam(self.netSTGAN1.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_G2 = torch.optim.Adam(self.netSTGAN2.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_D  = torch.optim.Adam([{'params': self.netSTGAN1.netD.parameters()},
                                                  {'params': self.netSTGAN2.netD.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G1, self.optimizer_G2, self.optimizer_D]
   
    def set_input(self, input):
        BaseModel.set_input(self, input)    
    
    def forward(self):
        # Compute output of generator 1
        self.inputSTGAN1 = torch.cat((self.input_img, self.skin_mask), 1) if self.opt.use_skinmask else self.input_img
        self.fake_shadow_image = self.netSTGAN1.forward_G(self.inputSTGAN1)
        
        # Compute output of generator 2
        self.inputSTGAN2 = torch.cat((self.input_img, self.fake_shadow_image, self.skin_mask), 1) if self.opt.use_skinmask else torch.cat((self.input_img, self.fake_shadow_image), 1)
        self.fake_free_shadow_image = self.netSTGAN2.forward_G(self.inputSTGAN2)

    def forward_D(self):
        fake_AB = torch.cat((self.inputSTGAN1, self.fake_shadow_image), 1)
        real_AB = torch.cat((self.inputSTGAN1, self.shadow_mask), 1)                                                            
        self.pred_fake, self.pred_real = self.netSTGAN1.forward_D(fake_AB.detach(), real_AB)
                                                         
        fake_ABC = torch.cat((fake_AB, self.fake_free_shadow_image), 1)
        real_ABC = torch.cat((real_AB, self.shadowfree_img), 1)                                                   
        self.pred_fake2, self.pred_real2 = self.netSTGAN2.forward_D(fake_ABC.detach(), real_ABC)
        
    def backward_D(self):        
        self.loss_D1_fake = self.GAN_loss(self.pred_fake, target_is_real = 0) 
        self.loss_D1_real = self.GAN_loss(self.pred_real, target_is_real = 1) 
        self.loss_D2_fake = self.GAN_loss(self.pred_fake2, target_is_real = 0) 
        self.loss_D2_real = self.GAN_loss(self.pred_real2, target_is_real = 1) 
        
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real)*0.1
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real)*0.1
        self.loss_D = self.loss_D1 + self.loss_D2*5
        self.loss_D.backward()
                                                            
    def backward_G1(self):
        self.loss_G1_GAN = self.GAN_loss(self.pred_fake, target_is_real = 1)                                           
        self.loss_G1_L1 = self.criterionL1(self.fake_shadow_image, self.shadow_mask)
        self.loss_G1 = self.loss_G1_GAN + self.loss_G1_L1*0.1
        self.loss_G1.backward(retain_graph=False)

    def backward_G2(self):
        self.loss_G2_GAN = self.GAN_loss(self.pred_fake2, target_is_real = 1)       
        self.loss_G2_L1 = self.criterionL1(self.fake_free_shadow_image, self.shadowfree_img)
        
        self.loss_G2 = self.loss_G2_GAN + self.loss_G2_L1*0.1
        self.loss_G2.backward(retain_graph=True)
        
    def get_prediction(self, input_img, skin_mask = None):
        BaseModel.get_prediction(self, input_img, skin_mask)
        return self.result
    
    def optimize_parameters(self):
        self.forward()
        
        self.set_requires_grad([self.netSTGAN1.netD, self.netSTGAN2.netD], True)  # Enable backprop for D1, D2
        self.forward_D()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad([self.netSTGAN1.netD, self.netSTGAN2.netD], False) # Freeze D
        self.forward_D()   
        
        self.optimizer_G2.zero_grad()
        self.backward_G2()
        self.optimizer_G2.step()
             
        self.optimizer_G1.zero_grad()
        self.backward_G1()
        self.optimizer_G1.step()

 

        
