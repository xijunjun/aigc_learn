"""For generating the data for training VQGAN, No polishing"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
# from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from basicsr.metrics import calculate_niqe
import math,cv2
import copy

from basicsr.utils import DiffJPEG
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import numpy as np


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


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

def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst


def get_imkey_ext(impath):
    imname=os.path.basename(impath)
    ext='.'+imname.split('.')[-1]
    imkey=imname[0:len(imname)-len(ext)]
    return imkey,ext

def makedir(dirtp):
    if os.path.exists(dirtp):
        return
    os.makedirs(dirtp)



if __name__=='__main__':
    
    device='cuda:0'
    imgsize=1024
    cfgvq=r'/disks/disk1/Workspace/mycode/algotrain/StableSR-main/configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml'
    vqgan_ckpt=r'/disks/disk1/Workspace/mycode/algotrain/StableSR-main/checkpoints/face_vqgan_cfw_00011.ckpt'
    imroot=r'/home/tao/mynas/Dataset/face_lq_test/testhat_1024'
    
    
    dstroot=imroot+'_resultvae'
    makedir(dstroot)
    
    ims=get_ims(imroot)
    
    
    vqgan_config = OmegaConf.load(cfgvq)
    vq_model = load_model_from_config(vqgan_config, vqgan_ckpt)
    vq_model = vq_model.to(device)
    
    
    
    
    vq_model.decoder.fusion_w = 0.0
    
    
    
    
    
    transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(imgsize),
		torchvision.transforms.CenterCrop(imgsize),
	])

        
    with torch.no_grad():
        
        for im in ims:
            
            imname=os.path.basename(im)
            imkey,ext=get_imkey_ext(im)
            cur_image = load_img(im).to(device)
            cur_image = transform(cur_image)
            cur_image = cur_image.clamp(-1, 1)
            
            img=cv2.imread(im)
            img=cv2.resize(img,(imgsize,imgsize))
            
        

            init_latent_generator, enc_fea_lq = vq_model.encode(cur_image)
            zin=init_latent_generator.sample()
            x_samples = vq_model.decode(zin, enc_fea_lq)
        
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * rearrange(x_samples[0].detach().cpu().numpy(), 'c h w -> h w c')
            x_sample =x_sample.astype(np.uint8)
            x_sample=cv2.cvtColor(x_sample,cv2.COLOR_RGB2BGR)
        

            cv2.imwrite(os.path.join(dstroot,imkey+'.jpg'),img)
            cv2.imwrite(os.path.join(dstroot,imkey+'_vae.jpg'),x_sample)

    
    
    
    print('finish')



