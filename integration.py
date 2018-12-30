import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import _visualize

import argparse
import json
import time
import csv

if __name__ == '__main__':
    # matplotlib.use('Agg')
    parser = argparse.ArgumentParser(
        description='Integrated counting detector')
    parser.add_argument('--folder', required=True,
                        metavar="/path/to/frames/folder",
                        help='Path to the folder containing frames to be labeled')
    parser.add_argument('--output_folder', required=False, default=None,
                        metavar="/path/to/store",
                        help='Folder to store labeled results')
    parser.add_argument('--model',  required=False,
                        default='/home/ye/diebold/detecting_counter/Mask_RCNN/models/best.h5',
                        metavar="/path/to/counting/weights.h5",
                        help='Path to counting detector weights')
    parser.add_argument('--num_gpus', required=False, default=1, type=int, help='Number of GPUs available')

    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = args.folder
    elif not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    class CountingConfig(coco.BagsConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        MEAN_PIXEL = [113.4660893, 108.65660642, 100.38982423]
        DETECTION_MIN_CONFIDENCE = 0.7
        DETECTION_NMS_THRESHOLD = 0.55
        RPN_ANCHOR_RATIOS = [0.333, 1, 2.25]
        RPN_ANCHOR_SCALES = [32, 64, 128, 256]
        RPN_NMS_THRESHOLD = 0.7

    class ClassConfig(coco.BagsConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        MEAN_PIXEL = [113.4660893, 108.65660642, 100.38982423]
        DETECTION_MIN_CONFIDENCE = 0.2
        DETECTION_NMS_THRESHOLD = 0.3
        RPN_ANCHOR_RATIOS = [0.333, 1, 3]
        RPN_ANCHOR_SCALES = (64, 128, 256, 512)
        RPN_NMS_THRESHOLD = 0.7
    config_counting = CountingConfig(1)
    config_class = ClassConfig(19)
    # config_counting.display()
    # config_class.display()

    # Create model object in inference mode.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_class)
    # Load weights trained on MS-COCO
    model.load_weights(args.model, by_name=True)
    
    class_names = ["None","Grill and chill Cooler Bag and 3pc bbq tool set",
    "Pebble True wireless earbuds",
    "20 oz stainless steel cross trainer water bottle",
    "Auto open folding umbrella",
    "2 Captains Chair",
    "Lime Green Roma Journal with phone pocket",
    "Duo Vacuum Stemless Wine Tumbler Gift Set",
    "Touch Screen Gloves In Pouch",
    "Arctic Zone® Ice Wall™ Lunch Cooler",
    "28-Can Coleman® Backpack Cooler",
    "Grey Wireless Speaker",
    "Sports jersey mesh drawstring backpack",
    "Bottles",
    "Lime green bag",
    "Tan/brown luggage tag",
    "Blue Journal book",
    "Green Earbuds",
    "Smartphone card holder ",
    "Grey Backpack",]
    print(len(class_names))
    class_names.sort()

    # start = time.time()
    # time_diff = 0
    # time_vis = 0

    # #for future use for generating json labels
    # imgs_json = []
    # labels_json = []
    # jid = 0
    img_names = os.listdir(args.folder)
    for name in img_names:
        if name.endswith('.jpg') or name.endswith('.png'):
            print(os.path.join(args.folder, name))
            image = skimage.io.imread(os.path.join(args.folder, name))
            # Run detection

            # height, width = image.shape
            # ct = image[1150:,:,: ]
            # cl = image[:1150,:,:]

            results = model.detect([image], verbose=0)
            # train_start = time.time()
            r = results[0]
            # train_end = time.time()
            # time_diff += train_end-train_start
            # vis_start = time.time()

            cur_score, cur_cls = visualize.save_instances(image, r, class_names, fname=os.path.join(args.output_folder, name), save=True)
            #frames_annotation.append((str(i).zfill(4), cur_cls, cur_score))
        #with open(os.path.join(args.output_folder, 'annotations.csv'), 'w') as of:
        #    writer = csv.writer(of)
        #    for item in frames_annotation:
        #        writer.writerow(item)
            
    print('FINISHED')
        # vis_end = time.time()
        # time_vis += vis_end - vis_start
        # # for future use for generating json labels
        # imgs_json.append({'file_name': name, 'id': i, 'height':image.shape[0], 'width':image.shape[1]})
        # for box in r_ct['rois']:
        #     labels_json.append({'image_id': i, 'bbox': box, 'category_id': 14, 'id': jid})
        #     jid += 1

    #{"segmentation": [[513, 174, 782, 174, 782, 705, 513, 705, 513, 174]], "area": 142839, "iscrowd": 0, "image_id": 0, "bbox": [513, 174, 782, 705], "category_id": 1, "id": 0}
    #
    # end = time.time()
    # print('total time %d'%((end-start)/60))
    # print('detection time %d'%(time_diff/100))
   # i prnd int('visualization time %d'%(time_vis/100))
