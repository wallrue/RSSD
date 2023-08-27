###############################################################################
# This file contains definitions of SID Net - Shadow Image Decomposition
# SIDNet is only to remove the shadow but detect shadow mask
###############################################################################

import torch
from torch import nn
from .network_GAN import define_G
import torch.nn.functional as F

class SIDNet(nn.Module):
    """ SIDNet includes G net and M net, which is to relit and remove shadow 
    from available shadow mask and full shadow image
    """
    def __init__(self, opt, net_g, net_m):
        super(SIDNet, self).__init__()
        #self.training = istrain    
        self.netG = define_G(opt.input_nc + 1 + opt.use_skinmask, 6, opt.ngf, net_g, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netM = define_G(6 + 1 + opt.use_skinmask, opt.output_nc, opt.ngf, net_m, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

    def forward(self, input_img, fake_shadow_image):
        self.input_img = F.interpolate(input_img,size=(256,256))
        self.fake_shadow_image = F.interpolate(fake_shadow_image,size=(256,256))
        inputG = torch.cat([self.input_img, self.fake_shadow_image], 1)
        # Compute output of generator 2
        self.shadow_param_pred = self.netG(inputG)
        
        w = inputG.shape[2]
        h = inputG.shape[3]
        n = self.shadow_param_pred.shape[0]
        
        # Compute lit image
        if len(self.shadow_param_pred.shape) > 2: 
            self.shadow_param_pred = torch.mean(self.shadow_param_pred.view([n,6,-1]),dim=2)

        add = self.shadow_param_pred[:,[0,2,4]]
        mul = (self.shadow_param_pred[:,[1,3,5]]*2) +3

        add = add.view(n,3,1,1).expand((n,3,w,h))
        mul = mul.view(n,3,1,1).expand((n,3,w,h))
        self.lit = self.input_img.clone()/2+0.5
        self.lit = self.lit*mul + add

        # Compute shadow matte
        inputM = torch.cat([self.input_img, self.lit, self.fake_shadow_image],1)
        self.alpha_pred = self.netM(inputM)
        self.alpha_pred = (self.alpha_pred +1) /2 
        
        # Compute free-shadow image
        self.final = (self.input_img/2+0.5)*(1-self.alpha_pred) + self.lit*(self.alpha_pred)
        self.final = torch.clamp(self.final*2-1, -1.0, 1.0)
        return self.shadow_param_pred, self.alpha_pred, self.final

def define_SID(opt, net_g = 'RESNEXT', net_m = 'unet_256'):
    net = SIDNet(opt, net_g, net_m)
    return net