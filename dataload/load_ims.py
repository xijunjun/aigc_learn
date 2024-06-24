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


def get_ims_curdir(imgpath):
    imgpathlst=[]
    for filename in os.listdir(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
            imgpathlst.append(os.path.join(imgpath, filename))
    return imgpathlst



def load_imnames(txtpath):
    imnames=[]
    with open(txtpath,'r') as f:
        lines=f.readlines() 
    for line in lines:
        imnames.append(line.rstrip('\n'))
    return imnames

def str2dict(strroot):
    lines=strroot.splitlines()
    
    dirdict=[]
    
    numkey=0
    for i,line in enumerate(lines):
        line=line.replace(' ','')
        if len(line)==0 or line.startswith('#'):
            continue
        print(i,line)
        
        if line.startswith('blacklist'):
            dirdict[numkey-1][1].append(line)
        else:
            # dirdict[line]=[]
            dirdict.append((line,[]))
            numkey+=1

    return dirdict        


def filter_imname(folder,blist,isintxt=True):
    
    namesall=[]
    for btxt in blist:
        namesall.extend(load_imnames(txtpath))
    ims=get_ims(folder)
    
    tpdict={}
    for im in ims:
        imname=os.path.basename(im)
        tpdict[imname]=0
    for imname in namesall:
        tpdict[imname]=0
    

    for im in ims:
        imname=os.path.basename(im)
        tpdict[imname]+=1

    for imname in namesall:
        tpdict[imname]+=2
    
    imsfilterd=[]
    for im in ims:
        imname=os.path.basename(im)
        if isintxt and tpdict[imname]==3:
            imsfilterd.append(im)
        if not isintxt and tpdict[imname]==1:
            imsfilterd.append(im)
    return imsfilterd
        

def loadims_from_dict(dirdict ,isintxt=True):
    ims=[]
    for i,item in enumerate( dirdict) :
        key,blist=item
        ims.extend(filter_imname(key,blist,isintxt=isintxt))
    return ims



if __name__=='__main__':
    
    lqroot=r'''
    /home/tao/mynas/Dataset/hairforsr/femalehd_crop2048
    blacklist:/home/tao/mynas/Dataset/hairforsr/notuse.txt
    # blacklist:/home/tao/mynas/Dataset/hairforsr/notuse2.txt
    /home/tao/mynas/Dataset/hairforsr/femalehd_crop2048
    blacklist:/home/tao/mynas/Dataset/hairforsr/notuse.txt
    # 1223
    '''
    
    txtpath=r'/home/tao/mynas/Dataset/hairforsr/notuse.txt'
    blacknames=load_imnames(txtpath)
    print(blacknames)
    
    
    dirdict=str2dict(lqroot)
    
    print('dirdict:',dirdict)
    
    ims=loadims_from_dict(dirdict,isintxt=False )
    
    print('len(ims):',len(ims))
    
    
    ims=get_ims_curdir(r'/home/tao/mynas/Dataset/hairforsr')
    
    print(ims)
    