# coding: utf-8
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import argparse
import json
import imutils
import glob

from config import Config

import model as modellib
import visualize_cv2 as visualize

NUM_CLS=22

class BagsConfig(Config):
    NAME = "bags"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + NUM_CLS  # background + classes
       
def intersects(r1, r2):
    return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy

def union(a,b):
    return (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])

def add_noise(typ,image,param):
     
    if typ == "gauss":
        
        mean = (0,0,0)
        std = (param, param, param) # 20 to 400
        gauss = image.copy()
        gauss = cv2.randn(gauss, mean, std)
        out = np.array(image.astype(int) + gauss.astype(int))
        out = out.clip(0,255).astype('uint8')
        
    elif typ == "sp":
        
        prob = param # 0.000005 to 0.2
        
        out = image.copy()
        thres = 1 - prob 
        
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    out[i][j] = [0,0,0]
                elif rdn > thres:
                    out[i][j] = [255,255,255]
        
    elif typ == "poisson":
    
        noise = np.random.poisson(image)
        out = image.astype(int)+param*noise.astype(int) # 0 to 7
        out = out.clip(0,255).astype('uint8')
        
    elif typ =="speckle":
        
         mean = (0,0,0) 
         std = (param,param,param) # 0.25 to 200
         noise = image.copy()
         noise = cv2.randn(noise,mean,std)
         out = image.astype(int) + image.astype(int)*noise.astype(int)
         out = out.clip(0,255).astype('uint8')
         
    return out


if __name__=='__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Demo Mask R-CNN.')
    parser.add_argument('--json_file', required=True,
                        metavar="/path/to/json_file/",
                        help='Path to JSON file')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--rotation', required=False,
                        help="Angle to rotate video input stream")
    parser.add_argument('--num_cls', required=True,
                        help="Number of classes in dataset without BG class")
    parser.add_argument('--video', required=False,
                        metavar="path/to/demo/video",
                        help='Video to play demo on', default=None)
    parser.add_argument('--image', required=False,
                        metavar="path/to/demo/image",
                        help='Video to play demo on')
    parser.add_argument('--image_dir', required=False,
                        metavar="path/to/demo/image/dir",
                        help='Image dir to play demo on')
    parser.add_argument('--show_upc', required=False, default=None,
                        help='Display UPC numbers instead of class_names from file')
    parser.add_argument('--save_demo', required=False, 
                        action='store_true', 
                        help='Saves demo to file instead of display')
    parser.add_argument('--resize', required=False,
                           help='Resizes demo for display by given fraction (0-1 input)')
    args = parser.parse_args()
    
    config = BagsConfig()
    config.display()

    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = args.model
    
    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
    
    cv2.namedWindow('frame',cv2.WND_PROP_FULLSCREEN)

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    class_names = ['BG'] + json.load(open(args.json_file))['classes']
    if hasattr(args, 'upc_file'):
        with open(args.upc_file, 'r') as f:
            for line in f.readlines():
                code, cls = [x.strip() for x in line.split(' ')]
                class_names[class_names.index(cls)] = code
    
    if args.video is not None:
        ret, image = cap.read()
        if args.resize is not None:
            image = cv2.resize(image, (0,0), fx=float(args.resize), fy=float(args.resize))
        out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (image.shape[1], image.shape[0]))
    elif args.image_dir is not None:
        img_files=glob.glob(os.path.join(args.image_dir, '*'))
        ret = True
    else:
        image = cv2.imread(args.image)
        ret = False      
        out=None
    ind = 0
    jnd = -1
    
    #noise_param = np.logspace(np.log10(0.0005), np.log10(0.2)) # salt and pepper
    #noise_param = np.logspace(np.log10(20), np.log10(400)) # gaussian
    #noise_param = np.logspace(np.log10(1), np.log10(7)) # poisson
    #noise_param = np.logspace(np.log10(0.25), np.log10(200)) # speckle
    noise_param = np.ones(50)*0.001
    np.savetxt('noise_params.txt', noise_param)
    
    while (1):
        if args.video is not None:
            ret, image = cap.read()
        elif args.image_dir is not None:
            image = cv2.imread(img_files[ind])
            ind+=1
            out = None
        else:
            image = cv2.imread(args.image)
            ret = True
        if args.resize is not None:
            image = cv2.resize(image, (0,0), fx=float(args.resize), fy=float(args.resize))
        if not ret:
            break

        if args.rotation is not None:
            image = imutils.rotate_bound(image, int(args.rotation))
        
        if jnd != -1:
            image = add_noise("sp", image, noise_param[jnd])
        jnd = jnd + 1
        
        if jnd == len(noise_param):
            break
        
        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        
        t = 'image' if args.image is not None else 'video'
        
        bboxes1, cls1, scores1 = [[x[0], x[1], x[2], x[3]] for x in r['rois']], r['class_ids'], r['scores']
        
        def iou_filter(bboxes, scores, cls, iou_threshold = 0.5):
            
            rboxes, rscores, rlabels, ignore, inter = [], [], [], [], False
            for i in range(len(bboxes)):
                if i in ignore:
                     continue
                for j in range(i+1, len(bboxes)):
                    if intersects(bboxes[i], bboxes[j]):
                        iou = area(bboxes[i], bboxes[j])/float(union(bboxes[i], bboxes[j])-area(bboxes[i], bboxes[j]))
                        if scores[i]<scores[j] and iou>iou_threshold:
                            inter = True
                        elif scores[i]>scores[j] and iou>iou_threshold:
                            ignore.append(j)
                
                if not inter:                
                    rboxes.append(bboxes[i])
                    rscores.append(scores[i])
                    rlabels.append(cls[i])
                else:
                    inter = False
            return rboxes, rlabels, rscores
        
        bboxes, classes, scores =iou_filter(bboxes1, scores1, cls1)
        classes = [int(x) for x in classes]
        '''
        image = imutils.rotate_bound(image, 270)
        results = model.detect([image], verbose=1)
        r = results[0]
        bboxes2, cls2, scores2 = [[x[0], x[1], x[2], x[3]] for x in r['rois']], r['class_ids'], r['scores']
        
        
        powerbank_index = class_names.index('Cape Three Power Bank')
        
        unique, counts = np.unique(cls1, return_counts=True)
        actual_powerbanks = counts[unique==powerbank_index]
        
        actual_powerbanks = 0 if len(actual_powerbanks)==0 else actual_powerbanks[0] 
        
        unique, counts = np.unique(cls2, return_counts=True)
        rotated_powerbanks = counts[unique==powerbank_index]
        rotated_powerbanks = 0 if len(rotated_powerbanks)==0 else rotated_powerbanks[0] 
        
        if actual_powerbanks < rotated_powerbanks:
            for _ in range(rotated_powerbanks - actual_powerbanks):
                cls1 = np.append(cls1,powerbank_index)
                scores1 = np.append(scores1, '0.99') #placeholder values
                bboxes1.append([0,0,5,5]) #placeholder values to be changed for visualization if necessary
        image = imutils.rotate_bound(image, 90)
        '''
        visualize.display_instances(image, np.array(bboxes), None, classes, 
                                    class_names, scores, save=args.save_demo, writer=out, ind=jnd)
        
        if args.image is not None and not args.save_demo or args.image_dir is not None:
            cv2.waitKey()
            
    if args.video is not None:
        cap.release()
        out.release()
    cv2.destroyAllWindows()
