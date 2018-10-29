# Wire_breakage_detector

## Introduction
This is the  code of the detector of wire's breakage which is one of the common defects in power system. This is pure Python code which can be treated as a simply introduction code to image processing, and can be easily transformed to other object detector.

## Pipeline
1. Read image from list.
2. Image segmentation. Give three common segmentation method: manual threold, OTSU algorithm and edge detection. Then ensemble the segmentation result to convert gray image to binary image.
3. Line detection. Using Hough transform line detection algorithm to find the location of each wire.
4. Generate Sliding window on each of lines. Just like `Proposal` method in object detection. 
5. Find the breakage location. Calculate the entropy of image to find the abnormity point. 
![image](https://github.com/gd2016229035/Wire_breakage_detector/blob/master/Figure_1.png)

## Software:
- python3.6
- python-opencv
- skimage
- matplotlib

## Run
`python main.py`

## Todo
- It takes about 30 seconds to detect one image which need to do speed optimization. Mainly because the DENSE line result in step3 cause to much sliding windows in step4.
