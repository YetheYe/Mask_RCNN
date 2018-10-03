import os
import sys
import random
import math
import cv2
import json
import numpy as np
from config import Config

import model as modellib

class BagsConfig(Config):
    def __init__(self, n, m=1, config=None):    
        
        if config:
            self.IMAGES_PER_GPU = 1
            self.IMAGE_MIN_DIM = config["IMAGE_MIN_DIM"]
            self.IMAGE_MAX_DIM = config["IMAGE_MAX_DIM"]
            self.IMAGE_PADDING = config["IMAGE_PADDING"]
            self.TRAIN_ROIS_PER_IMAGE = config["TRAIN_ROIS_PER_IMAGE"]
            self.ROI_POSITIVE_RATIO = config["ROI_POSITIVE_RATIO"]
            self.MEAN_PIXEL = np.array(config["MEAN_PIXEL"])
            self.BACKBONE= config["BACKBONE"]

            self.USE_MINI_MASK = True
            self.MAX_GT_INSTANCES = config["MAX_GT_INSTANCES"]
        else:
            self.IMAGES_PER_GPU = 1

        self.NAME = "qick"
        self.NUM_CLASSES = 1 + n 
        self.GPU_COUNT = m
        
        super().__init__()

class hans2:

    def __init__(self, json_file, model_path, con):
        
        with open(json_file, 'r') as f:
            classes = json.load(f)['classes']

        config = BagsConfig(n=len(classes), m=1, config=con)
        
        config.display()

        self.model = modellib.MaskRCNN(mode="inference", model_dir=os.path.dirname(model_path), config=config)
        self.model.load_weights(model_path, by_name=True)

        self.class_names = [ 'BG' ] + json.load(open(json_file))['classes']


    def return_objects(self, image):
        # ## Run Object Detection
        labels = self.model.detect([image], verbose=0)

        labels = labels[0]

        results = []
        for i in range(len(labels['class_ids'])):
            results.append({
                'class_idx': labels['class_ids'][i],
                'class_name': self.class_names[labels['class_ids'][i]],
                'score': labels['scores'][i],
                'roi': labels['rois'][i],
                'mask': labels['masks'][i]
            })

        return results

if __name__ == '__main__':
    ai = hans2('model/CISCO/cisco_new.json', 'model/CISCO/cisco_mask_rcnn_csv.h5')
    #print(ai.return_objects('1top1.png'))
