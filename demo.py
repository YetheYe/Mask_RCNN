# coding: utf-8

# # Mask R-CNN Demo

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

from config import Config

import model as modellib
import visualize_cv2 as visualize

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
    parser.add_argument('--trim', required=False,
                        help="Remove possible black bars if any due to rotation")
    parser.add_argument('--num_cls', required=True,
                        help="Number of classes in dataset without BG class")
    parser.add_argument('--video', required=False,
                        metavar="path/to/demo/video",
                        help='Video to play demo on')
    parser.add_argument('--image', required=False,
                        metavar="path/to/demo/image",
                        help='Video to play demo on')
    parser.add_argument('--show_upc', required=False,
                        help='Display UPC numbers instead of class_names from file')
    parser.add_argument('--save_demo', required=False, 
                        action='store_true', 
                        help='Saves demo to file instead of display')
    args = parser.parse_args()

    class BagsConfig(Config):
        """Configuration for training on MS COCO.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """

        # Give the configuration a recognizable name
        NAME = "bags"

        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + int(args.num_cls)  # background [index: 0] + 1 person class tranfer from COCO [index: 1] + 12 classes

    config = BagsConfig()
    config.display()

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = args.model
    # Download COCO trained weights from Releases if needed
    
    if not hasattr(args, 'image'):
    	# Directory of images to run detection on
    	cap = cv2.VideoCapture(args.video)

    # ## Configurations
    # 
    # We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
    # 
    # For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

    # In[2]:



    # ## Create Model and Load Trained Weights

    # In[3]:


    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # ## Class Names
    # 
    # The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
    # 
    # To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
    # 
    # To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
    # ```
    # # Load COCO dataset
    # dataset = coco.CocoDataset()
    # dataset.load_coco(COCO_DIR, "train")
    # dataset.prepare()
    # 
    # # Print class names
    # print(dataset.class_names)
    # ```
    # 
    # We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

    # In[4]:


    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG'] + json.load(open(args.json_file))['classes']
    if hasattr(args, 'upc_file'):
        with open(args.upc_file, 'r') as f:
            for line in f.readlines():
                code, cls = [x.strip() for x in line.split(' ')]
                class_names[class_names.index(cls)] = code
    
    # ## Run Object Detection
    
    if not hasattr(args, 'image'):
    	# In[5]:
    	ret, image = cap.read()
    else:
    	image = cv2.imread(args.image)
    if args.rotation is not None:
        image = imutils.rotate_bound(image, int(args.rotation))
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (image.shape[1], image.shape[0]))        

    if args.trim:
        image = trim(image)

    while (1):
        if not hasattr(args, 'image'):
            # Load a random image from the images folder
            ret, image = cap.read()
        else:
            image = cv2.imread(args.image)
        if not ret:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if args.rotation is not None:
            image = imutils.rotate(image, int(args.rotation))

        if args.trim:
            image = trim(image)
            
        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
        #                            class_names, r['scores'], display=False, writer=out)
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'], save=args.save_demo, writer=out)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
