###############################################################################
# This file contains transform methods which will be used to transform images 
# when loading dataset in dataset classes
###############################################################################

import random
import numbers
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import cv2

def get_transform_list(opt):
    """The function for calling transform methods by command ('resize_and_crop', 
    'resize', 'scale_width_and_crop', ...)
    """
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop': 
        # Resize to loadSize and crop to fineSize
        transform_list.append(Resize(opt.loadSize))
        transform_list.append(RandomCrop(newSize = opt.fineSize))
    elif opt.resize_or_crop == 'resize': 
        # Resize to fineSize
        transform_list.append(Resize(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width': 
        # Scale to fineSize for width only
        transform_list.append(Scale(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width_and_crop': 
        # Scale to fineSize for width only and crop to fineSize
        transform_list.append(Scale(opt.loadSize))
        transform_list.append(RandomCrop(newSize = opt.fineSize))
    elif opt.resize_or_crop == 'none': 
        # Just modify the width and height to be multiple of 4
        pass
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)

    # Horizontal Flip Option
    if opt.isTrain:
        if not opt.no_flip:
            transform_list.append(RandomHorizontalFlip())
            
    # Finish the transform list
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize(0.5, 0.5)]
    
    return transform_list

def get_transform_for_synthetic(opt, dataset_name):
    """The function for calling transform methods for raw synthetic dataset
    """
    imageSize = opt.fineSize
    scale_ratio = random.random()*0.5

    if dataset_name == 'handimg': 
        transform_list = [Scale(imageSize + int(imageSize*scale_ratio)),
                          RandomRotate(),
                          CenterCrop(imageSize), 
                          #RandomFlip(), 
                          transforms.ToTensor(),
                          transforms.Normalize(0.5, 0.5),
                          enhance_hand()]
    elif dataset_name == 'background': 
        transform_list = [Scale(imageSize),
                          transforms.ToTensor(),
                          transforms.Normalize(0.5, 0.5)]
    elif dataset_name == 'shadow': 
        transform_list = [Scale(imageSize + int(imageSize*scale_ratio)),
                          RandomRotate(),
                          CenterCrop(imageSize), 
                          transforms.ToTensor(),
                          transforms.Normalize(0.5, 0.5),
                          enhance_shadow()]
    else:
        raise ValueError('--dataset_name does not exist')
        
    return transform_list



class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, newSize, padding=0):
        self.newSize = (int(newSize), int(newSize)) if isinstance(newSize, numbers.Number) else newSize
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
       
        w, h = img.size
        th, tw = self.newSize
        if w == tw and h == th:
            output = img
        else: 
            x1 = random.randint(0,max(0,w-tw-1))
            y1 =  random.randint(0,max(0,h-th-1))
            output = img.crop((x1, y1, x1 + tw, y1 + th))
        return output
    
class CenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        (w, h) = img.size
        (th, tw) = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))
    
class RandomRotate(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img):
        return img.rotate(random.random() * 360, Image.Resampling.NEAREST, expand=1, fillcolor=(255, 255, 255))
    
class Resize(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.Resampling.NEAREST):
        self.size = (int(size), int(size)) if isinstance(size, numbers.Number) else size
        self.interpolation = interpolation

    def __call__(self, img):
        output = img.resize(self.size, self.interpolation)
        return output
    
class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.Resampling.NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            output = img
        elif w < h:
            ow = self.size
            oh = int(self.size * h / w)    
            output = img.resize((ow, oh), self.interpolation)   
        else:
            oh = self.size
            ow = int(self.size * w / h)
            output = img.resize((ow, oh), self.interpolation)
        return output

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""
    def __call__(self, img):
        flag = random.random() < 0.5
        if flag:
            output = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            output = img
        return output
    
class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        tensor = (tensor - self.mean)/self.std
        return tensor  
  
## Functions for synthetic dataset
def get_binary(img):  # input dim = 4
    img = (img.clone() + 1.0) / 2.0 
    r = img[0]*255.0
    g = img[1]*255.0
    b = img[2]*255.0

    img_gray  = (.299*r + .587*g + .114*b)/255.
    img_gray = np.uint8(img_gray.numpy() * 255.0)
    
    (ret2, th2) = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY) # + cv2.THRESH_OTSU)
    th2 = cv2.bitwise_not(th2)
    
    out_temp = np.float32(th2) * 2.0 / 255.0 - 1.0
    hand_shape = torch.tensor(out_temp)
    return hand_shape

class enhance_hand(object):
    def __call__(self, imgs):
        tensor = imgs
        binar = get_binary(tensor.clone())
        morph_size = 9
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
        hbinar = cv2.morphologyEx(binar.numpy(), cv2.MORPH_CLOSE, element)
        return {'img': tensor,
                'binary_normal': torch.from_numpy(-hbinar)[None, :, :], # Invert binary image
                'binary_mask': torch.from_numpy(hbinar)[None, :, :] }
    
class enhance_shadow(object):
    def __call__(self, imgs):
        tensor = imgs
        binar = -get_binary(tensor.clone())
        enhance_shadow = cv2.GaussianBlur(binar.numpy(), (75,75), cv2.BORDER_DEFAULT)
        return torch.from_numpy(enhance_shadow)[None, :, :]