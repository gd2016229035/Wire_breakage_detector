# coding=UTF-8
from config import opt
from utils import *
from preprocess import *
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

def line_dect(img, img_gray, img_avg):
    img_1 = copy.deepcopy(img)
    lines = cv2.HoughLinesP(img_avg,1,np.pi/180,5,minLineLength=1000,maxLineGap=50)
    lines1 = lines[:,0,:]#提取为二维
    for x1,y1,x2,y2 in lines1[:]: 
        cv2.line(img_1,(x1,y1),(x2,y2),(0,0,255),5)
    # lines = cv2.HoughLines(img_avg,1,np.pi/180,500)
    # lines1 = lines[:,0,:]#提取为为二维
    # for rho,theta in lines1[:]: 
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a)) 
    #     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

#    box_show(img)

    return lines1, img_1

def move_box(img1, img_gray, lines, flag):
    #img, img_gray, lines = line_dect()
    img = copy.deepcopy(img1)
    size = img.shape
    p_w = opt.w_ratio * size[1]
    p_h = opt.h_ratio * size[0]
    alph = opt.alph
    lines_num = lines.shape[0]
    j = 0
    output = [[] for i in range(lines_num)]

    for line in lines:
        
        point_x1, point_y1, point_x2, point_y2 = line
        point_x1 = float(point_x1)
        point_y1 = float(point_y1)
        point_x2 = float(point_x2)
        point_y2 = float(point_y2)

        k = (point_y2 - point_y1)/(point_x2 - point_x1)
        num = int((point_x2 - point_x1)/(alph * p_w)) + 2 # move more two boxes

        if abs(k) > 1: #steep line
            # skip very steep lines(not real lines)
            if abs(k) > 10:
                break

            x1 = int(point_x1)
            if k < 0:       
                y1 = int(point_y1 + p_w * k)
            else:
                y1 = int(point_y1)     
                       
            w = p_w
            h = abs(p_w * k)
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            for i in range(num):
                if flag:
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
                im_crop1 = img_gray[y1:y2, x1:x2]
                output[j].append([x1, y1, x2, y2])
                output[j].append(Imentropy(im_crop1))
                x1 += int(alph * p_w)
                #cross the image border
                if x1+int(alph * p_w) >= size[1]:
                    break
                y1 += int(alph * k * p_w)
                x2 = int(x1 + w)
                y2 = int(y1 + h)
        #flat line        
        else:
            x1 = int(point_x1)
            y1 = int(point_y1 - p_h/2)
            w = p_w
            h = p_h 
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            for i in range(num):
                if flag:
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
                im_crop1 = img_gray[y1:y2, x1:x2]
                output[j].append([x1, y1, x2, y2])
                output[j].append(Imentropy(im_crop1))
                x1 += int(alph * p_w)
                #cross the image border
                if x1+int(alph * p_w) >= size[1]:
                    break
                y1 += int(alph * k * p_w)
                x2 = int(x1 + w)
                y2 = int(y1 + h)

        j += 1
    return output

def choose_breakbox(img1, output):
    img = copy.deepcopy(img1)
    break_bbox_all = np.array([[0, 0, 0, 0]]) 
    for line in output:
        bbox = np.array(line)[0::2]
        Imentropy = np.array(line)[1::2]
        Imentropy_no_null = Imentropy[Imentropy > 0]# drop the 0 Imentropy
        if len(Imentropy_no_null) == 0: # skip the null Imentropy array 
            continue
        th1 = Imentropy_no_null.mean()
        break_bbox = [bbox[i] for i in range(len(Imentropy)) if ((Imentropy[i] > 1.1 * th1) or (Imentropy[i] < th1 / 1.3)) and Imentropy[i] > 0]
        if break_bbox != []:
            break_bbox_all = np.append(break_bbox_all, np.array(break_bbox), axis=0)
   
    bbox_nms = non_max_suppression_fast(break_bbox_all[1:], overlapThresh = opt.nms_thresh)
    for break_one in bbox_nms:
        cv2.rectangle(img,(break_one[0],break_one[1]),(break_one[2],break_one[3]),(255,0,0),5)
    return img
    #imgshow(img)

    #plt.imshow(img,)
    #plt.show()