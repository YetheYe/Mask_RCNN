from PIL import Image
import glob
import numpy as np
import cv2
import math
from cv2 import moveWindow
from scipy import ndimage

count = 0

for file_name in glob.glob('bags/*.jpg'):

    count+=1

    im = Image.open(file_name)
    sorte = im.getcolors(im.size[0]*im.size[1])
    sorte.sort(reverse=True, key= lambda x: x[0])

    one = np.array(sorte[0][1])
    two = np.array(sorte[1][1])
    dif = sum(abs(one-two))
    
    segment = np.zeros((np.shape(np.array(im))[0], np.shape(np.array(im))[1]))

    for n, col in sorte:
        if (sum(abs(one-col))>dif):
            two = col
            dif = sum(abs(one-col))
    
    if(dif<65):
        continue
        
    if(sum(abs(one-np.array(im)[0][0]))<sum(abs(two-np.array(im)[0][0]))):
        temp = one
        one = two
        two = temp
    
    min_x, min_y, max_x, max_y = np.shape(np.array(im))[0], np.shape(np.array(im))[1], 0, 0
    
    for key1, vals in enumerate(np.array(im)):
        for key2, rgb in enumerate(vals):
            if(sum(abs(rgb-one))<700):
                segment[key1][key2]=255
                if(key1<min_x):
                    min_x = key1
                if(key2<min_y):
                    min_y = key2
                if(key1>max_x):
                    max_x = key1
                if(key2>max_y):
                    max_y = key2
    
    
    np_im = np.array(im)
    cv2.rectangle(np_im, (int(min_y), int(max_x)), (int(max_y), int(min_x)), (255,0,0), 5)
    print ((int(min_x), int(max_y)), (int(max_x), int(min_y)))
    
    #segment = segment[math.ceil(b1[1]):math.ceil(b2[1]), math.ceil(b1[0]):math.ceil(b2[0])]

    #cv2.namedWindow("first")
    #cv2.namedWindow("second")

    cv2.imwrite('Outputs/bag_%d_bbox.png'%(count),np_im[...,::-1])
    cv2.imwrite('Outputs/bag_%d_mask.png'%(count),segment)

    #img = cv2.imshow("first", segment)
    #moveWindow("second", 500, 500)

    #img = cv2.imshow("second", np_im)
    
