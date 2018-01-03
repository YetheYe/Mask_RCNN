from PIL import Image
import glob
import numpy as np
import cv2
import os
import math
from cv2 import moveWindow
from scipy import ndimage
from collections import Counter
import webcolors

from matplotlib import patches
import matplotlib.pyplot as plt


def find_bbox(file_name):

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
    
    for key1, vals in enumerate(np.array(im)):
        for key2, rgb in enumerate(vals):
            if(sum(abs(rgb-one))<700):
                if(key1<min_x):
                    min_x = key1
                if(key2<min_y):
                    min_y = key2
                if(key1>max_x):
                    max_x = key1
                if(key2>max_y):
                    max_y = key2
    return min_y, max_y, min_x, max_x

def find_mask(file_name, shapes=None):

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
    
    segments = []
    
    shapes = [['dummy_class', [0, im.size[0], 0, im.size[1]]]] if shapes is None else shapes
    
    for _, bbox in shapes:
        
        segment = np.zeros((np.shape(np.array(im))[0], np.shape(np.array(im))[1]))
        
        for key1, vals in enumerate(np.array(im)[bbox[0]:bbox[1]]):
            for key2, rgb in enumerate(vals[bbox[2]:bbox[3]]):
                if(sum(abs(rgb-one))<700):
                    segment[key1+bbox[0]][key2+bbox[2]]=1
        segments.append(segment)            
    
    return segments

