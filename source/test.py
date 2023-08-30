###############################################################################
# This file is used for testing trained models. 
# There are three major parameters in main function which needs being defined: 
# - dataset_dir: the root folder which contains dataset (with the datasetname)
# - checkpoints_dir: the folder to save checkpoints after training
# - training_dict: the model to be trained (with the datasetname)
# dataset is trained with models in training_dict in a run
###############################################################################

import sys
import os
import torch
import numpy as np
import time
import cv2
from PIL import Image
from torch.autograd import Variable
from options.test_options import TestOptions
from data import CustomDatasetDataLoader
from models import create_model
from models.loss_function import calculate_ssim, calculate_psnr
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #Fix error on computer
  
def progressbar(it, info_dict, size=60, out=sys.stdout):
    count = len(it)
    def show(j, batch_size):
        n = batch_size*j if batch_size*j < count else count
        x = int(size*n/count)
        taken_time = time.time() - info_dict["start time"]
        print("\r[{}{}] {}/{} | {:.3f} secs".format("#"*x, "."*(size-x), n, count, taken_time), 
                end='', file=out, flush=True) # Flushing for progressing bar in Python 3.0 
        sys.stdout.flush() # Flushing for progressing bar in Python 2.0 
        
    show(0, 1)
    for i, item in enumerate(it):
        yield i, item
        if i == 0: # Initialize batch_size value
            batch_size = len(list(item.values())[0])
        show(i+1, batch_size)
    print("", flush=True, file=out) # Do thing after ending iteration
      
def print_current_losses(log_dir, model, losses, t_comp):
    message = '{\"testing model\": \"%s\", \"computing time\": %.3f' % (model, t_comp)
    for k, v in losses.items():
        message += ', \t\"%s\": %s' % (k, v)
    message += '}'

    print(" - Result of testing : " + message)  # Print the message
    with open(log_dir, "a+") as log_file:
        log_file.write('%s\n' % message)  # Save the message

def GetSkinMask(tensor_img): #Tensor (4 channels) in range [-1, 1]   
    result_list = list()
    for i in range(len(tensor_img)):
        num_img = tensor_img[i].cpu()
        num_img = (np.transpose(num_img, (1,2,0)) + 1.0)/2.0
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
        result_list.append(global_mask)
    result = torch.from_numpy(np.array(result_list))
    return result

def evaluate(dataset, test_model, result_dir, folder_name, opt):
    cuda_tensor = torch.cuda.FloatTensor if len(opt.gpu_ids) > 0 else torch.FloatTensor
    PNSR_dict = {"original": 0.0, "shadowmask": 0.0, "shadowfree": 0.0}
    SSIM_dict = {"original": 0.0, "shadowmask": 0.0, "shadowfree": 0.0}
    
    path_list = [os.path.join(result_dir,"original"), 
                 os.path.join(result_dir,"groudtruth"),
                 os.path.join(result_dir,"shadowmask_{}_{}".format(folder_name, opt.epoch)),  
                 os.path.join(result_dir,"shadowfree_{}_{}".format(folder_name, opt.epoch))]
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)
    
    tcomp = 0
    progressbar_info = {"start time": time.time()}
    for _, data in progressbar(dataset, progressbar_info):
        list_imgname = data['imgname']
        real_shadowfull = Variable(data['shadowfull'].type(cuda_tensor), 0)
        real_shadowmask = Variable(data['shadowmask'].type(cuda_tensor), 0)
        real_shadowfree = Variable(data['shadowfree'].type(cuda_tensor), 0)

        real_shadowfull = torch.reshape(real_shadowfull,(-1,3,256,256))
        real_shadowmask = torch.reshape(real_shadowmask,(-1,1,256,256))
        real_shadowfree = torch.reshape(real_shadowfree,(-1,3,256,256))
        
        testing_start_time = time.time()   
        if not opt.use_skinmask:
            prediction = test_model.get_prediction(real_shadowfull)
        else:
            skin_mask = (GetSkinMask(real_shadowfull) >0)*2-1
            skin_mask = torch.reshape(skin_mask,(-1,1,256,256))
            prediction = test_model.get_prediction(real_shadowfull, skin_mask)
        tcomp += time.time() - testing_start_time
                 
        fake_shadowfree = prediction['final']
        fake_shadowmask = prediction['phase1']
        
        for i in range(len(real_shadowfull)):
            img_shadowfull = real_shadowfull[i].permute(1, 2, 0)
            img_shadowmask = real_shadowmask[i][0]
            img_shadowfree = real_shadowfree[i].permute(1, 2, 0)
            pre_shadowmask = fake_shadowmask[i][0].data
            pre_shadowfree = fake_shadowfree[i].data.permute(1, 2, 0)

            PNSR_dict["original"] += calculate_psnr(img_shadowfree, img_shadowfull)
            PNSR_dict["shadowmask"] += calculate_psnr(img_shadowmask, pre_shadowmask)
            PNSR_dict["shadowfree"] += calculate_psnr(img_shadowfree, pre_shadowfree)

            SSIM_dict["original"] += calculate_ssim(img_shadowfree, img_shadowfull)
            SSIM_dict["shadowmask"] += calculate_ssim(img_shadowmask, pre_shadowmask)
            SSIM_dict["shadowfree"] += calculate_ssim(img_shadowfree, pre_shadowfree)
            # Save result of processing to list
            result_list = list()
            result_list.append(((img_shadowfull + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            result_list.append(((img_shadowfree + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            result_list.append(((pre_shadowmask + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            result_list.append(((pre_shadowfree + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            
            # Save output image after processing
            for idx, path in enumerate(path_list):
                data_name = os.path.join(path,"{}".format(list_imgname[i]))
                if not os.path.isfile(data_name):
                    result_img = result_list[idx]
                    if len(np.shape(result_img)) == 3 and np.shape(result_img)[2] == 3:
                        Image.fromarray(result_img).convert('RGB').resize((224, 224)).save(data_name)
                    elif len(np.shape(result_img)) == 2:
                        Image.fromarray(result_img).resize((224, 224)).save(data_name)
      
    length = len(dataset)
    PNSR_dict = {k: v / length for k, v in PNSR_dict.items()}
    SSIM_dict = {k: v / length for k, v in SSIM_dict.items()}
    return PNSR_dict, SSIM_dict, tcomp
        
if __name__=='__main__':
    """The main function for testing model. There are three important parameters:
    - dataset_dir: the root folder which contains dataset. {datasetname: path}
    - checkpoints_dir: the folder to save checkpoints after training. {datasetname: path}
    - training_dict: the model to be trained {datasetname: modelname}
    Example of datasetname: shadowparam, shadowsynthetic, single
    Example of modelname: DSDSID, SIDSTGAN, STGAN
    """
    checkpoint_dir = os.path.join(os.getcwd(), "_checkpoints")      
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    test_options = TestOptions()
    dataset_dir = os.path.join(os.path.join(os.getcwd(),"_database"),"NTUST_HS_Testset")

    testing_dict = [["SIDPAMIwISTGAN",  True],
                    ["SIDSTGAN",        True],
                    ["STGAN",           True],
                    ["SIDPAMIwISTGAN",  False],
                    ["SIDSTGAN",        False],
                    ["STGAN",           False],
                    ]
        
    result_dir = os.path.join(os.getcwd(), "_result_set")
    for model_name, use_skinmask in testing_dict:    
        print('============== Start testing: model {} =============='.format(model_name))
        # Model defination   
        test_options.dataset_mode = "rawsynthetic"
        test_options.checkpoints_root = checkpoint_dir  
        test_options.model_name = model_name
        opt = test_options.parse()
        
        opt.use_skinmask = use_skinmask        
        if opt.use_skinmask:
            opt.name = opt.name + "_HandSeg"
        test_options.print_options(opt)
        
        model = create_model(opt)
        model.setup(opt)

        # Loading test dataset
        opt.dataset_mode = "NTUST_HS"
        opt.dataroot = dataset_dir
        data_loader = CustomDatasetDataLoader(opt)
        dataset = data_loader.load_data()

        experiment_name = opt.name
        PNSR_score, SSIM_score, computing_time = evaluate(dataset, model, result_dir, experiment_name, opt)
        #print_current_losses(os.path.join(result_dir, 'valid.log'), experiment_name, {"PNSR_score": PNSR_score, "SSIM_score": SSIM_score}, computing_time)
