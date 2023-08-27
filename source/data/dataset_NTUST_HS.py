###############################################################################
# This file contains class of the dataset named NTUSTHSDataset which includes 
# full shadow images, free shadow images and params of shadow 
# this dataset is used to test data
###############################################################################

import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset
from data.transform import get_transform_list
from data.image_folder import make_dataset
import cv2

class NTUSTHSDataset(BaseDataset):
    def name(self):
        return 'NTUSTHSDataset'
    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'train_shadowfull')
        self.dir_C = os.path.join(opt.dataroot, 'train_shadowfree')
        
        self.A_paths, self.imname = make_dataset(self.dir_A)
        self.A_size = len(self.A_paths)
        
        self.transformData = transforms.Compose(get_transform_list(opt))
     
    def __getitem__(self,index):
        birdy = dict()
        
        index_A = index % self.A_size
        imname = self.imname[index_A]
        A_path = self.A_paths[index_A]
        
        A_img = self.transformData(Image.open(A_path).convert('RGB'))  
        w, h = A_img.size()[0], A_img.size()[1]
        B_img =  self.transformData(Image.fromarray(np.zeros((int(w),int(h)),dtype = np.float),mode='L'))
        C_img = self.transformData(Image.open(os.path.join(self.dir_C, imname)).convert('RGB'))
   
        # Finishing package of dataset information 
        birdy['shadowfull'] = A_img
        birdy['shadowmask'] = B_img
        birdy['shadowfree'] = C_img
        
        A_img_num = (np.transpose(birdy['shadowfull'].numpy(), (1,2,0)) + 1.0)/2.0 
        skinmask = self.GetSkinMask(A_img_num)
        skinmask = torch.from_numpy(skinmask*2.0 -1.0)
        birdy['skinmask'] = torch.permute(skinmask, (2, 0, 1))
        
        birdy['imgname'] = imname
        birdy['w'] = w
        birdy['h'] = h
        birdy['shadowfull_paths'] = A_path
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
        return self.A_size
