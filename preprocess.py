# coding=UTF-8
from config import opt
from utils import imgshow

import cv2
import os
from skimage.morphology import disk
import skimage.filters.rank as sfr
import numpy as np

#img_name = (opt.file_list)[0]


def read_image(img_name):
    img = os.path.join(opt.path, img_name)
    BGRimg = cv2.imread(img, -1)
    img=cv2.cvtColor(BGRimg, cv2.COLOR_BGR2RGB)
    size = img.shape
    scale = opt.resize_sale
    img_resize = cv2.resize(img, (int(size[1]*scale), int(size[0]*scale)), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
    return img_resize, img_gray

#DUCO_RemoveBackGround ref: https://blog.csdn.net/XiaoHeiBlack/article/details/53106087?locationNum=2&fps=1
def precrocessing(img_gray, size):
    bk = sfr.minimum(img_gray, disk(size))#min 
    bk = cv2.blur(bk,(size,size)) #mean
    newIm = img_gray - bk
    #dst = cv2.blur(dst1, (2,2))
    return newIm

def manual_thresh(img_gray):#阈值化处理
    (T, thresh)= cv2.threshold(img_gray, opt.threshold_min, opt.threshold_max, cv2.THRESH_BINARY)
    return thresh

def OTSU_thresh(img_gray):
    ret2,th2 = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2

def edge_thresh(img_gray):
    canny = cv2.Canny(img_gray,100,200)
    sp = canny.shape
    canny_inv = np.zeros((sp[0],sp[1]),dtype=np.uint8) 
    canny_inv[canny==0] = 255
    canny_inv[canny==255] = 0
    return canny_inv

def afterprocessing(img_th):
    kernel = np.ones((5,5),np.uint8)    
    erosion = cv2.erode(img_th, kernel, iterations = 1)  
    dilation = cv2.dilate(erosion, kernel, iterations = 1) 
    #closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)   # imfill ?
    return dilation

def segment(img_gray1):
    #img_rgb, img_gray1 = read_image(img)
    img_gray = precrocessing(img_gray1 ,10)

    img_th1 = manual_thresh(img_gray)
    img_th1 = afterprocessing(img_th1)

    img_th2 = OTSU_thresh(img_gray)
    img_th2 = afterprocessing(img_th2)

    img_th3 = edge_thresh(img_gray)
    img_th3 = afterprocessing(img_th3)

    ## ensemble result
    #img_avg = (img_th1 + img_th2 + img_th3)/3
    img_avg = (img_th1 + img_th2)/2 
    img_avg = img_avg.astype(np.uint8)

    img_avg = OTSU_thresh(img_avg)
    return img_avg
    #return img_avg

#imgshow(th2,'gray')


