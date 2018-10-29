# coding=UTF-8
from config import opt
from utils import *
from preprocess import *
from lines import *
import time

filelist = opt.file_list

def test():
    i = 1
    for img in filelist:
        # 1. Load image from list
        img_rgb, img_gray = read_image(img)
        # 2. Convert to binary image through image segmentation 
        img_avg = segment(img_gray)
        # 3. Line detection
        lines, img_line = line_dect(img_rgb, img_gray, img_avg)
        # 4. Use Sliding window on the line
        output = move_box(img_line, img_gray, lines, opt.show_move_box) 
        # 5. Calculate the 'broken' place
        result_img = choose_breakbox(img_rgb, output)
        # 6. Show the result
        sub_imgshow(i, img_rgb, img_avg, img_line, result_img)

        i += 1

if __name__ == '__main__':
    test()
    print('===== finish ======')



