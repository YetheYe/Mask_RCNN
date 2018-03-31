import os
import sys
import random
import math
import cv2
import json

from config import Config

import model as modellib

class BagsConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "bags"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 12  # background [index: 0] + 1 person class tranfer from COCO [index: 1] + 12 classes

def return_objects(img_path, json_file, model_path):

    config = BagsConfig()
    
    image = cv2.imread(img_path)
    model = modellib.MaskRCNN(mode="inference", model_dir=os.path.dirname(model_path), config=config)

    model.load_weights(model_path, by_name=True)
    
    class_names = ['BG'] + json.load(open(json_file))['classes']
    
    # ## Run Object Detection
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.detect([image], verbose=0)

    # Visualize results
    r = results[0]
        
    return [class_names[i] for i in r['class_ids']]

if __name__ == '__main__':
    print (return_objects('1top1.png', 'model/pascal_dataset.json', 'model/mask_rcnn_bags_0005.h5'))
