import os,sys
import torch
from utils_yaml import *
# from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
# from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
# from basicsr.data.transforms import paired_random_crop
from utils_general import *
from PIL import Image
import PIL
from einops import rearrange, repeat
import  math
# from basicsr.utils import DiffJPEG, USMSharp
# from basicsr.utils.img_process_util import filter2D
from torch.nn import functional as F
import time

from funcs_basicsr import *
from degrade_resr_official import DegradeResrOfficial



import numpy as np
import torch



def tensor2cv(tensor):
    return np.transpose(tensor.cpu().detach().numpy(), (0, 2, 3, 1)).squeeze()

def cv2tensor(cv_mat):
    return torch.from_numpy(np.expand_dims(np.transpose(cv_mat.astype(np.float32), (2, 0, 1)), axis=0))



def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.


def pil2cv2(pil_image):
    # 将 PIL 图像转换为 NumPy 数组
    rgb_image = np.array(pil_image)
    # 将 RGB 转换为 BGR
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def load_img_rgb_0_1_tensor(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return image


def load_img_bg_0_1_cv(path,isbgr=True):
    image = cv2.imread(path)
    if isbgr: 
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image).astype(np.float32) / 255.0
    return image

def imgcv2tensor(image):
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image
    
    

def rgb_0_1_tensor2cv(imagein):

    image=255*rearrange(imagein[0].detach().cpu().numpy(), 'c h w -> h w c')
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image.astype(np.uint8)



# rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
# 


def imgtrans_cv(opt,gt):
    gt = augment(gt, opt['use_hflip'], opt['use_rot'])
    return gt



if __name__=='__main__':

    opt=yaml_load(r'/disks/disk1/Workspace/mycode/gitproj/aigc_learn/image_degrade/train_realesrgan_x4plus.yml')

    imroot=r'/home/tao/mynas/Dataset/hairforsr/ultrahd'


    optdot=DotDict(opt)

    ims=get_ims(imroot)


    degrade_obj=DegradeResrOfficial(opt)



    for im in ims:

        # imgcv=cv2.imread(im)
        while 1:
        
            image_tensor=load_img_rgb_0_1_tensor(im)
            print(image_tensor.shape)


            imgcv=load_img_bg_0_1_cv(im)

            # imgcv=imgtrans_cv(opt,imgcv)


            imgcv=cv2.resize(imgcv,(1024,1024))

            image_tensor=imgcv2tensor(imgcv)


            time_st=time.time()
            pred_tensor=degrade_obj.run_data(image_tensor)
            runtime=time.time()-time_st
            
            print('runtime:',runtime)


            # imgcv=rgb_0_1_tensor2cv(pred_tensor)

            imgcv=(tensor2cv(pred_tensor)*255).astype(np.uint8)
            imgcv=cv2.cvtColor(imgcv, cv2.COLOR_RGB2BGR)
            

            print(im)


            cv2.imshow('img', limit_img_auto(imgcv))

            if cv2.waitKey(20)==27:
                exit(0)




    print(opt['datasets'])
    print(optdot.datasets.train)





    print('finish')



