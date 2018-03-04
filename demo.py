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

from config import Config

import model as modellib
import visualize_cv2 as visualize

class BagsConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "bags"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 20  # background [index: 0] + 1 person class tranfer from COCO [index: 1] + 12 classes

if __name__=='__main__':

    config = BagsConfig()
    config.display()


    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = sys.argv[2]
    # Download COCO trained weights from Releases if needed

    # Directory of images to run detection on
    cap = cv2.VideoCapture(sys.argv[1])

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
    class_names = ('BG', 'blue_perfume', 'black_perfume', 'double_speedstick', 'blue_speedstick', 'dove_blue', 'dove_perfume', 'dove_pink', 'green_speedstick', 'gear_deo', 'dove_black', 'grey_speedstick', 'choc_blue', 'choc_red', 'choc_yellow', 'black_cup', 'nyu_cup', 'ilny_white', 'ilny_blue', 'ilny_black', 'human')
            

    # ## Run Object Detection

    # In[5]:

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1080,1920))

    while (1):
        # Load a random image from the images folder
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
        #                            class_names, r['scores'], display=False, writer=out)
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
    cap.release()
    out.release()
