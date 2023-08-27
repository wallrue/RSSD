###############################################################################
# This file contains loss functions used to create the model for training
###############################################################################

import math
import cv2
import numpy as np

def calculate_smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    loss = 0
    weight = 1.

    dx, dy = gradient(pred_map)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
    return loss

def calculate_psnr(img1_tensor, img2_tensor):
    """Calculate PSNR score
    
    Parameters:
        img1_tensor (int) -- a image to compare, range [0, 255]
        img2_tensor (int) -- a image to compare, range [0, 255]
    """    
    img1 = ((img1_tensor + 1.0)*255.0/2.0).cpu().numpy().astype(np.float64)
    img2 = ((img2_tensor + 1.0)*255.0/2.0).cpu().numpy().astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1_tensor, img2_tensor):
    """Calculate SSIM score
    
    Parameters:
        img1_tensor (int) -- a image to compare, range [0, 255]
        img2_tensor (int) -- a image to compare, range [0, 255]
    """
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    img1 = ((img1_tensor + 1.0)*255.0/2.0).cpu().numpy().astype(np.float64)
    img2 = ((img2_tensor + 1.0)*255.0/2.0).cpu().numpy().astype(np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')