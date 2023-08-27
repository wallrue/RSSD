###############################################################################
# This file contains definitions of Stacked Conditional GAN (STGAN)
###############################################################################

import torch
from torch import nn
from .network_GAN import define_G, define_D, GANLoss

class STGANNet(nn.Module):
    """ STGAN is built from two GANs. This class is the definition of a 
    single GAN architecture, which includes a generator (Unet32) and 
    a discriminator (PatchGAN discriminator)
    """
    def __init__(self, opt, gan_input_nc, gan_output_nc, net_g, net_d):
        super(STGANNet, self).__init__()
        self.netG = define_G(gan_input_nc, gan_output_nc, opt.ngf, net_g, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netD = define_D(gan_input_nc+gan_output_nc, opt.ngf, net_d, 3, opt.norm, 
                                         True, opt.init_type, opt.init_gain, opt.gpu_ids)
        
        self.GAN_loss = GANLoss(opt.gpu_ids)
        self.criterionL1 = torch.nn.L1Loss().to(1)
        
    def forward_G(self, input_img):
        self.fake_image = self.netG(input_img)
        return self.fake_image
    
    def forward_D(self, fake_package, real_package): 
        self.pred_fake = self.netD(fake_package)
        self.pred_real = self.netD(real_package)
        return self.pred_fake, self.pred_real
      
def define_STGAN(opt, gan_input_nc, gan_output_nc, net_g = 'unet_32', net_d = 'n_layers'):
    net = STGANNet(opt, gan_input_nc, gan_output_nc, net_g, net_d)
    return net

