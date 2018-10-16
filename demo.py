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

class BagsConfig(Config):
    NAME = "bags"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE='resnet101'   
    
    def __init__(self, n):
        self.NUM_CLASSES = 1 + n  # background + classes
        super().__init__()        

def iou_filter(bboxes, scores, cls, class_names, iou_threshold=0.2):

    def intersects(r1, r2):
        return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

    def area(a, b):  # returns None if rectangles don't intersect
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        if (dx>=0) and (dy>=0):
            return dx*dy

    def union(a,b):
        return (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])

    return_str = ""
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
            return_str.append(class_names[cls[i]]+" (%.5f)\n"%scores[i])
            inter = False
    return np.array(rboxes), np.array(rlabels), np.array(rscores), return_str

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
    
    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    
    with open(args.json_file, 'r') as f:
        obj = json.load(f)
        
    config = BagsConfig(len(obj['classes']))
    config.display()

    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = args.model
    
    if args.video is not None:
        cap = cv2.VideoCapture(args.video)

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
    while (1):
        if args.video is not None:
            ret, image = cap.read()
        elif args.image_dir is not None:
            if ind == len(img_files):
                break
            image = cv2.imread(img_files[ind])
            ind+=1
            out = None
        else:
            image = cv2.imread(args.image)
            ret = not ret    
        if args.resize is not None:
            image = cv2.resize(image, (0,0), fx=float(args.resize), fy=float(args.resize))
        if not ret:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if args.rotation is not None:
            image = imutils.rotate_bound(image, int(args.rotation))

        # Run detection
        results = model.detect([image], verbose=1)
        # Visualize results
        r = results[0]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        t = 'image' if args.image is not None else 'video'
        
        r['rois'], r['class_ids'], r['scores'], ret_str = iou_filter(r['rois'], r['scores'], r['class_ids'], class_names)
        
        rois, cls, scr = [], [], []
        for R, C, S in zip(r['rois'], r['class_ids'], r['scores']):
            if (R[2]+R[0])<2*275:
                rois.append(R)
                cls.append(C)
                scr.append(S)
        
        r['rois'], r['class_ids'], r['scores'] = rois, cls, scr

        cut, crois, cscores, cclasses = 970, [], [], []
        for (roi, cls), score in zip(zip(r['rois'], r['class_ids']), r['scores']):
            
            if roi[1] > cut or cls==6: # remove illuminati random detection
                #ret_str += "%s (%.5f)\n"%(class_names[cls], score)
                continue
            else:
                crois.append(roi)
                cscores.append(score)
                cclasses.append(cls)
        
        r['rois'], r['class_ids'], r['scores'] = np.array(crois), np.array(cclasses), np.array(cscores)

        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'], ind=1, save=img_files[ind-1] if args.save_demo else None, writer=out, dtype='image', softmax=ret_str)
        if args.image is not None and not args.save_demo or args.image_dir is not None:
            c = cv2.waitKey(1)
            if args.image_dir is not None:
                if c==81:
                    ind-=2
    if args.video is not None:
        cap.release()
        out.release()
    cv2.destroyAllWindows()
