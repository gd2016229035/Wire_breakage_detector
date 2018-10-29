# coding=UTF-8
class Config:
    #path = '../easy/'
    path = './images/'
    # file_list=['太和F4太草干#11杆1(080716).JPG', '太和F4太草干#11杆2(080716).JPG', '太和F4太草干#11杆3(080716).JPG',\
    # '太和F4太草干#11杆(080716).JPG' ,'DSCN1226.JPG',  'IMAG1251.jpg',\
    # 'IMG_1193.JPG', 'IMG_2220.JPG', 'IMG_20171114_144835.jpg']
    file_list=['demo.JPG']

    show_move_box = False

    w_ratio = 0.09
    h_ratio = 0.09
    alph = 0.5

    resize_sale = 0.8
    
    threshold_min=100
    threshold_max=200

    nms_thresh = 0.3

opt = Config()