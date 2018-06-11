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
    
    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    
    config = BagsConfig()
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
        img_files=
    else:
        image = cv2.imread(args.image)
        ret = False      
        out=None
    ind = 0
    while (1):
        if args.video is not None:
            ret, image = cap.read()
        elif args.image_dir is not None:
            image = cv2.imread(cv2.imread(
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
        
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'], save=args.save_demo, writer=out, dtype=t)
        if args.image is not None and not args.save_demo:
            cv2.waitKey()
    if args.video is not None:
        cap.release()
        out.release()
    cv2.destroyAllWindows()
