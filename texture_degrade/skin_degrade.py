
import os,sys
import torch
import cv2
import numpy as  np



def limit_img_auto(imgin):
    img=np.array(imgin)
    sw = 1920 * 1.2
    sh = 1080 * 1.2
    h, w = tuple(list(imgin.shape)[0:2])
    swhratio = 1.0 * sw / sh
    whratio = 1.0 * w / h
    resize_ratio=sh/h
    if whratio > swhratio:
        resize_ratio=1.0*sw/w
    if resize_ratio<1:
        img=cv2.resize(imgin,None,fx=resize_ratio,fy=resize_ratio)
    return img


def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst


def skindegrade(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)

    result = cv2.inpaint(image, (adaptive_thresh == 0).astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA)


    return result


if __name__=='__main__':

    imroot=r'Z:\Dataset\phone\facehd\skinlabel'

    ims=get_ims(imroot)

    for im in ims:
        img=cv2.imread(im)
        img=cv2.resize(img,None,fx=0.5,fy=0.5)


        imgd=skindegrade(img)


        cv2.imshow('img',limit_img_auto(img))

        cv2.imshow('imgd',limit_img_auto(imgd))


        if cv2.waitKey(0)==27:
            exit(0)





    print('finish')