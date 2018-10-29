import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import numpy as np  

def imgshow(img):
    plt.figure()
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img,cmap ='gray') 
    plt.show() 

def sub_imgshow(i, img_rgb, img_avg, img_line, result_img):
    fig = plt.figure(i)
    #plt.ion()
    ax = fig.add_subplot(2,2,1)
    ax.set_title('RGB image')
    ax.imshow(img_rgb)
    ax = fig.add_subplot(2,2,2)
    ax.set_title('Segment image')
    ax.imshow(img_avg, cmap ='gray')
    ax = fig.add_subplot(2,2,3)
    ax.set_title('Line detect')
    ax.imshow(img_line) 
    ax = fig.add_subplot(2,2,4)
    ax.set_title('Result img')
    ax.imshow(result_img)
    # plt.ioff()
    plt.show() 


def Imentropy(img):
    size = img.shape
    tmp = []  
    for i in range(size[0]*size[1]):  
        tmp.append(0)  
    val = 0  
    k = 0  
    res = 0 
    for i in range(len(img)):  
        for j in range(len(img[i])):  
            val = img[i][j]  
            tmp[val] = float(tmp[val] + 1)  
            k =  float(k + 1)  
    for i in range(len(tmp)):  
        tmp[i] = float(tmp[i] / k)  
    for i in range(len(tmp)):  
        if(tmp[i] == 0):  
            res = res  
        else:  
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))  
    return res


def non_max_suppression_fast(boxes, overlapThresh):  
    # if there are no boxes, return an empty list  
    if len(boxes) == 0:  
        return []  
   
    # if the bounding boxes integers, convert them to floats --  
    # this is important since we'll be doing a bunch of divisions  
    if boxes.dtype.kind == "i":  
        boxes = boxes.astype("float")  
   
    # initialize the list of picked indexes   
    pick = []  
   
    # grab the coordinates of the bounding boxes  
    x1 = boxes[:,0]  
    y1 = boxes[:,1]  
    x2 = boxes[:,2]  
    y2 = boxes[:,3]  
   
    # compute the area of the bounding boxes and sort the bounding  
    # boxes by the bottom-right y-coordinate of the bounding box  
    area = (x2 - x1 + 1) * (y2 - y1 + 1)  
    idxs = np.argsort(y2)  
   
    # keep looping while some indexes still remain in the indexes  
    # list  
    while len(idxs) > 0:  
        # grab the last index in the indexes list and add the  
        # index value to the list of picked indexes  
        last = len(idxs) - 1  
        i = idxs[last]  
        pick.append(i)  
   
        # find the largest (x, y) coordinates for the start of  
        # the bounding box and the smallest (x, y) coordinates  
        # for the end of the bounding box  
        xx1 = np.maximum(x1[i], x1[idxs[:last]])  
        yy1 = np.maximum(y1[i], y1[idxs[:last]])  
        xx2 = np.minimum(x2[i], x2[idxs[:last]])  
        yy2 = np.minimum(y2[i], y2[idxs[:last]])  
   
        # compute the width and height of the bounding box  
        w = np.maximum(0, xx2 - xx1 + 1)  
        h = np.maximum(0, yy2 - yy1 + 1)  
   
        # compute the ratio of overlap  
        overlap = (w * h) / area[idxs[:last]]  
   
        # delete all indexes from the index list that have  
        idxs = np.delete(idxs, np.concatenate(([last],  
            np.where(overlap > overlapThresh)[0])))  
   
    # return only the bounding boxes that were picked using the  
    # integer data type  
    return boxes[pick].astype("int")  