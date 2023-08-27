###############################################################################
# This file contains class of the dataset named RawSyntheticDataset which includes 
# hand image, back ground and shadow images; in addition, NTUST_IP
# those subsets will be combined together in order to generate 
# a synthetic dataset each training loop
###############################################################################

import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import random
import cv2
from imutils import paths
from PIL import Image
from scipy.optimize import curve_fit
from data.base_dataset import BaseDataset
from data.transform import get_transform_for_synthetic
from scipy.special import binom

### -------------- Random Shape Function --------------------------------------

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)
def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)    

def random_shape_generate():
    rad = 0.2
    edgy = 0.05
    x,y, _ = get_bezier_curve(get_random_points(n=7, scale=1), rad=rad, edgy=edgy)
    images_point = np.zeros((256, 256, 3), np.uint8)
    points = np.vstack((x, y))
    points = np.transpose(points*255).astype('int')
    
    images_point = cv2.fillPoly(images_point, pts =[points], color=(255,255,255))
    return Image.fromarray(images_point)


### -------------- RawSyntheticDataset --------------------------------------


class RawSyntheticDataset(BaseDataset):
    def name(self):
        return 'RawSyntheticDataset'
    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_shadowmask = os.path.join(opt.dataroot, 'shadow')
        self.dir_background = os.path.join(os.path.join(opt.dataroot, 'background'),'val')
        self.dir_handimg = os.path.join(opt.dataroot, 'hands')
        
        self.shadow_list = list(paths.list_images(self.dir_shadowmask))
        self.background_list = list(paths.list_images(self.dir_background))
        self.handimg_list = list(paths.list_images(self.dir_handimg))
        
        random.shuffle(self.shadow_list)
        random.shuffle(self.background_list)
        random.shuffle(self.handimg_list)

        self.transformData_handimg = transforms.Compose(get_transform_for_synthetic(self.opt, 'handimg'))
        self.transformData_background = transforms.Compose(get_transform_for_synthetic(self.opt, 'background'))
        self.transformData_shadow = transforms.Compose(get_transform_for_synthetic(self.opt, 'shadow'))
        
        self.count = 0 #Variables to save samples from dataset
        self.using_generated_shadow = True
        self.saved_folder = os.path.join(os.getcwd(), '_train_samples')
        if not os.path.exists(self.saved_folder):
            os.mkdir(self.saved_folder)
            
        # Adding the NTUST_IP dataset ---------------------------------------
        self.dir_handimg_ip = os.path.join(opt.dataroot, 'NTUST_IP')
        self.adding_NTUST_IP = os.path.exists(self.dir_handimg_ip) #True
        if self.adding_NTUST_IP:
            self.handimg_ip_list = list(paths.list_images(self.dir_handimg_ip))
            random.shuffle(self.handimg_ip_list)
            

    def shadow_validator(self, shadow_img, handmask_img):
        valid_score = np.sum(shadow_img)/ np.sum(handmask_img) 
        return valid_score > 0.2 and valid_score < 0.65
    
    def concaten_hand(self, background, norm_mask, object_img): # Concatenate Hands to Background
        # Combine background and hand
        r1 = background[:,:,0]*norm_mask[:,:,0] + np.where(norm_mask[:,:,0] == 0.0, object_img[:,:,0], 0.0)
        r2 = background[:,:,1]*norm_mask[:,:,0] + np.where(norm_mask[:,:,0] == 0.0, object_img[:,:,1], 0.0)
        r3 = background[:,:,2]*norm_mask[:,:,0] + np.where(norm_mask[:,:,0] == 0.0, object_img[:,:,2], 0.0)
        result_img = np.array([r1,r2,r3])
        return np.transpose(result_img, (1,2,0))
    
    def concaten_shadow(self, background, shadow_img): # Concatenate Hands to Background 
        if True: #random.random() > 0.5:
                beta = -1.0+random.betavariate(3,5) + 0.1 
                beta1, beta2, beta3 = beta - random.random()*0.2, beta - random.random()*0.2, beta - random.random()*0.2
        else:
                beta1 = beta2 = beta3 = -(random.random() % ((95 - 30) + 1) + 30)/100.0; #random from -0.3 to -0.95
        
        r1 = background[:,:,0] + shadow_img[:,:,0]*beta1 
        r2 = background[:,:,1] + shadow_img[:,:,0]*beta2
        r3 = background[:,:,2] + shadow_img[:,:,0]*beta3 
        result_img = np.array([r1,r2,r3])
        result_img = np.transpose(result_img, (1,2,0))
        np.clip(result_img, 0.0, 1.0, out=result_img)
        return result_img
    
    def shadow_on_hand(self, hand_mask, shadow_img): # Concatenate Hands to Background
        hand_shadeless = hand_mask*(1.0-shadow_img)
        hand_shaded    = hand_mask*shadow_img
        
        np.clip(hand_shadeless, 0.0, 1.0, out=hand_shadeless)
        np.clip(hand_shaded, 0.0, 1.0, out=hand_shaded)
        return hand_shadeless, hand_shaded

    def relit(self, x, a, b): # Functions for computing relit param
        return np.uint8((a * x.astype(np.float64)/255 + b)*255)
    
    def im_relit(self, Rpopt,Gpopt,Bpopt,dump): # Functions for computing relit param
        #some weird bugs with python
        sdim = dump.copy()
        sdim.setflags(write=1)
        sdim = sdim.astype(np.float64)
        sdim[:,:,0] = (sdim[:,:,0]/255) * Rpopt[0] + Rpopt[1]
        sdim[:,:,1] = (sdim[:,:,1]/255) * Gpopt[0] + Gpopt[1]
        sdim[:,:,2] = (sdim[:,:,2]/255) * Bpopt[0] + Bpopt[1]
        sdim = np.uint8(sdim*255)
        return sdim

    def compute_params(self, shadowfull_image, shadowmask_image, shadowfree_image): # Functions for computing relit param
        kernel = np.ones((5,5),np.uint8)
        
        sd = np.uint8(shadowfull_image*255.0)
        #mean_sdim = np.mean(sd, axis=2)
        
        mask_ori = np.uint8(shadowmask_image*255.0)
        mask = cv2.erode(mask_ori, kernel, iterations = 2)
        
        sdfree = np.uint8(shadowfree_image*255.0) 
        #mean_sdfreeim = np.mean(sdfree, axis=2)

        #i, j = np.where(np.logical_and(np.logical_and(np.logical_and(mask>=1,mean_sdim>5),mean_sdfreeim<230),np.abs(mean_sdim-mean_sdfreeim)>10))
        i, j = np.where(mask>=0)
      
        source = sd*0
        source[tuple([i,j])] = sd[tuple([i,j])] 
        target = sd*0
        target[tuple([i,j])]= sdfree[tuple([i,j])]
        
        R_s = source[:,:,0][tuple([i,j])]
        G_s = source[:,:,1][tuple([i,j])]
        B_s = source[:,:,2][tuple([i,j])]
        
        R_t = target[:,:,0][tuple([i,j])]
        G_t = target[:,:,1][tuple([i,j])]
        B_t = target[:,:,2][tuple([i,j])]
        
        c_bounds = [[1,-0.1],[10,0.5]]
        Rpopt, pcov = curve_fit(self.relit, R_s, R_t, bounds=c_bounds)
        Gpopt, pcov = curve_fit(self.relit, G_s, G_t, bounds=c_bounds)
        Bpopt, pcov = curve_fit(self.relit, B_s, B_t, bounds=c_bounds)
        
        return Rpopt[1],Rpopt[0],Gpopt[1],Gpopt[0],Bpopt[1],Bpopt[0]
        
    def __getitem__(self, index):
        
        birdy = dict()
        
        if index < len(self.handimg_list):
            index_img = index
            handimg = self.transformData_handimg(Image.open(self.handimg_list[index_img]).convert("RGB"))
            background = self.transformData_background(Image.open(self.background_list[index_img]).convert("RGB"))
            
            background_img = (np.transpose(background.numpy(), (1,2,0)) + 1.0)/2.0 
            hand_img = (np.transpose(handimg['img'].numpy(), (1,2,0)) + 1.0)/2.0
            hand_norm = (np.transpose(handimg['binary_normal'].numpy(), (1,2,0)) + 1.0)/2.0
            hand_mask = (np.transpose(handimg['binary_mask'].numpy(), (1,2,0)) + 1.0)/2.0
        else:         # Adding the NTUST_IP dataset ----------------------------------------
            index_img = index - len(self.handimg_list)
            background = self.transformData_background(Image.open(self.handimg_ip_list[index_img]).convert("RGB"))
            background_img = (np.transpose(background.numpy(), (1,2,0)) + 1.0)/2.0 
            hand_img = background_img
            hand_norm = torch.ones((np.shape(background_img)[0], np.shape(background_img)[1], 1)).numpy()
            hand_mask = torch.FloatTensor(self.GetSkinMask(background_img)).numpy() 
            
            background_mask = torch.ones((np.shape(background_img)[0], np.shape(background_img)[1], 1)).numpy()
            if np.sum(hand_mask)/ np.sum(background_mask) < 0.2:
                #If we can not detect hand mask, we will not filter shadow inside hand palm
                hand_mask = background_mask
                
        # Create shadow mask on hand
        if self.using_generated_shadow: #using
            shadowimg = random_shape_generate()
        else:
            shadow_id = index_img % len(self.shadow_list)
            shadowimg = Image.open(self.shadow_list[shadow_id]).convert("RGB")
        shadowimg = self.transformData_shadow(shadowimg)
        shadowimg = (np.transpose(shadowimg.numpy(), (1,2,0)) + 1.0)/2.0
        shadow_img = shadowimg*hand_mask
        
        count_while = 0
        while(not self.shadow_validator(shadow_img, hand_mask)):
            count_while += 1
            if count_while > 10: #Save 1 sample
                cv2.imwrite(os.path.join(self.saved_folder,"fail_handmask.png"), np.uint8(hand_mask*255))
                cv2.imwrite(os.path.join(self.saved_folder,"fail_shadowimg.png"), np.uint8(shadow_img*255))
                
            if self.using_generated_shadow: #using
                shadowimg = random_shape_generate()
            else: #using shadow_mask from ISTD dataset
                shadow_id = (shadow_id+1)%len(self.shadow_list)            
                shadowimg = Image.open(self.shadow_list[shadow_id]).convert("RGB")
            shadowimg = self.transformData_shadow(shadowimg)
            shadowimg = (np.transpose(shadowimg.numpy(), (1,2,0)) + 1.0)/2.0
            shadow_img = shadowimg*hand_mask
            
        shadow_img = cv2.GaussianBlur(shadow_img, (101,101), 11)
        shadow_img = np.expand_dims(shadow_img, axis=2)
        
        full_hand_img = self.concaten_hand(background_img, hand_norm, hand_img)
        full_shadow_img = self.concaten_shadow(full_hand_img, shadow_img)
        
        shadow_param = self.compute_params(full_shadow_img, shadow_img, full_hand_img)  
        skinmask = self.GetSkinMask(full_shadow_img)
        # Save dataset to review ----------------------------------------------
        if self.count < 1: #Save 1 sample
            self.count += 1
            
            cv2.imwrite(os.path.join(self.saved_folder,"{}_full_shadow_img.png".format(index)), cv2.cvtColor(np.uint8(full_shadow_img*255), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.saved_folder,"{}_shadow_img.png".format(index)), np.uint8(shadow_img*255))
            cv2.imwrite(os.path.join(self.saved_folder,"{}_full_hand_img.png".format(index)), cv2.cvtColor(np.uint8(full_hand_img*255), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.saved_folder,"{}_hand_img.png".format(index)), cv2.cvtColor(np.uint8(hand_img*255), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.saved_folder,"{}_hand_mask.png".format(index)), np.uint8(hand_mask*255))
            cv2.imwrite(os.path.join(self.saved_folder,"{}_skin_mask.png".format(index)), np.uint8(skinmask*255))
                        
        shadowfull_image = torch.from_numpy(full_shadow_img*2.0 -1.0)
        shadowmask_image = torch.from_numpy(shadow_img*2.0 -1.0)
        shadowfree_image = torch.from_numpy(full_hand_img*2.0 -1.0)
        handmask_img = torch.from_numpy(hand_mask*2.0 -1.0)
        skinmask_image = torch.from_numpy(skinmask*2.0 -1.0)
     
        image_size = shadowfull_image.size()
        # Finishing package of dataset information    
        birdy['shadowfull'] = torch.permute(shadowfull_image, (2, 0, 1))
        birdy['shadowmask'] = torch.permute(shadowmask_image, (2, 0, 1))
        birdy['shadowfree'] = torch.permute(shadowfree_image, (2, 0, 1))
        birdy['handmask'] = torch.permute(handmask_img, (2, 0, 1))
        birdy['w'] = image_size[0]
        birdy['h'] = image_size[1]

        birdy['shadowparams'] = torch.FloatTensor(np.array(shadow_param))
        birdy['skinmask'] = torch.permute(skinmask_image, (2, 0, 1))
        birdy['imgname'] = "img_{}.png".format(index_img)

        return birdy 
    
    def GetSkinMask(self, num_img): #Tensor (3 channels) in range [0, 1]   
        img = np.uint8(num_img*255)

        # Skin color range for hsv color space 
        img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((9,9), np.uint8))
        
        # Skin color range for hsv color space 
        img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((9,9), np.uint8))
        
        # Merge skin detection (YCbCr and hsv)
        global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
        global_mask=cv2.medianBlur(global_mask,3)
        global_mask = cv2.dilate(global_mask, np.ones((9,9), np.uint8), iterations = 2)
        
        global_mask = np.expand_dims(global_mask/255, axis=2)
        return global_mask
    
    def __len__(self):
        return len(self.handimg_list) + len(self.handimg_ip_list) if self.adding_NTUST_IP else len(self.handimg_list)
    
    
