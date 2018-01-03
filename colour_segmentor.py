from PIL import Image
import glob
import numpy as np
import cv2
import os
import math
from cv2 import moveWindow
from scipy import ndimage
from collections import Counter
import lxml.etree
import lxml.builder    
import time

from matplotlib import patches
import matplotlib.pyplot as plt

def find_object_bbox_masks(file_name, shapes=None):

    calculate_bbox = True
    bboxes, segments = None, None
    if (shapes is not None):
        bboxes = [bbox for _, bbox in shapes]
        calculate_bbox = False
    else:
        shapes = [[file_name.split('/')[-2], [0, 0, Image.open(file_name).size[1], Image.open(file_name).size[0]]]]
    annot_path = file_name[:-4]
    
    if (os.path.exists(annot_path+'_mask.npy') and os.path.exists(annot_path+'_bbox.npy')):
        segments = np.load(annot_path+'_mask.npy')
        bboxes = np.load(annot_path+'_bbox.npy')
    else:
        im = Image.open(file_name)
        sorte = im.getcolors(im.size[0]*im.size[1])
        sorte.sort(reverse=True, key= lambda x: x[0])

        one = np.array(sorte[0][1])
        two = np.array(sorte[1][1])
        dif = sum(abs(one-two))
        
        for n, col in sorte:
            if (sum(abs(one-col))>dif):
                two = col
                dif = sum(abs(one-col))
        
        if(dif<65):
            return None
            
        if(sum(abs(one-np.array(im)[0][0]))<sum(abs(two-np.array(im)[0][0]))):
            temp = one
            one = two
            two = temp
        
        min_x, min_y, max_x, max_y = np.shape(np.array(im))[0], np.shape(np.array(im))[1], 0, 0
        segments, bboxes = np.zeros((im.size[0], im.size[1], len(shapes))), []
        
        for i, (target_class, bbox) in enumerate(shapes):
            
            if (max(bbox)<1):
                if(len(shapes)==1):
                    shapes[0][1] = [0, 0, im.size[1], im.size[0]]
                else:
                    continue
            
            for key1, vals in enumerate(np.array(im)[int(bbox[0]):int(bbox[1])]):
                for key2, rgb in enumerate(vals[int(bbox[2]):int(bbox[3])]):
                    if(sum(abs(rgb-one))<700):
                        segments[key1+int(bbox[0])][key2+int(bbox[2])][i]=1
                        if (calculate_bbox):
                            if(key1+int(bbox[1])<min_x):
                                min_x = key1+int(bbox[1])
                            if(key2+int(bbox[0])<min_y):
                                min_y = key2+int(bbox[0])
                            if(key1+int(bbox[3])>max_x):
                                max_x = key1+int(bbox[3])
                            if(key2+int(bbox[2])>max_y):
                                max_y = key2+int(bbox[2])
            if (calculate_bbox):
                if (max_x>=im.size[0] or max([min_x, min_y, max_x, max_y])<1):
                    max_x = im.size[0]-1
                if (max_y>=im.size[1] or max([min_x, min_y, max_x, max_y])<1):
                    max_y = im.size[1]-1
            
            if (calculate_bbox):
                bboxes.append([min_x, min_y, max_x, max_y])
        
        if(not calculate_bbox):
            bboxes =  np.array([bbox for name, bbox in shapes])
        
        np.save(annot_path+'_mask', np.array(segments, dtype=bool))
        np.save(annot_path+'_bbox', np.array(bboxes, dtype=np.int32))
        
    return np.array(segments, dtype=np.int32), np.array(bboxes, dtype=np.int32)

def test_time():
    tic = time.time()

    seg, bb = find_object_bbox_masks('Data/bags/black_ameligalanti/2017-L7-CK2-20780452-01-1.jpg')

    print (time.time()-tic)

