#coding:utf-8
import cv2
import os
import numpy as np
import shutil,platform
import torch
from torch.nn import functional as F


def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst



if __name__=='__main__':
    imroot=r'/home/tao/mynas/Dataset/hairforsr/notuse'
    ims=get_ims(imroot)
    
    txtpath=imroot+'.txt'
    
    lines=''
    for im in ims:
        imname=os.path.basename(im)
        lines+=imname+'\n'
        
    
    lines=lines.rstrip('\n')
    
    with open(txtpath,'w') as f:
        f.writelines(lines)
    
    
    print('finish')
