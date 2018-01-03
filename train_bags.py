import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from colour_segmentor import find_bbox, find_object_masks

from config import Config
import utils
import model as modellib
import visualize
from model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class BagsConfig(Config):
    
    # Give the configuration a recognizable name
    NAME = "bags"

    GPU_COUNT = 3
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 12 + 1  # background [index: 0] + 1 person class tranfer from COCO [index: 1] + 12 classes
    STEPS_PER_EPOCH = 3000
    VALIDATION_STEPS = 100
    
config = BagsConfig()
config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class BagsDataset(utils.Dataset):
    """Generates the bags dataset. 
    """
    
    def load_bags(self, part):
        """
        part: train/eval
        """

        classes = ['black_backpack', 'nine_west_bag', 'meixuan_brown_handbag', 'sm_bdrew_grey_handbag', 'wine_red_handbag', 'sm_bclarre_blush_crossbody', 'mk_brown_wrislet', 'black_plain_bag', 'lmk_brown_messenger_bag', 'sm_peach_backpack', 'black_ameligalanti', 'white_bag']
        
        count = 0

        # Add classes
        
        self.add_class("bags", 1, "person")
        for i, c in enumerate(classes):
            self.add_class("bags", i+2, c)
        
        pattern = re.compile("bot[0-9]*.png")
        
        for images in glob.glob('Data/handbag_images/JPEGImages/'):
            
            shapes = []
            f = images.rsplit('/')[1]
            ann_path = images.split('JPEGImages')[0]+'Annotations/'+f[:-3]+'xml'
                
            tree = ET.parse(ann_path)
            root = tree.getroot()
	         
	        img_path = 'handbag_images/'+root.find('path').text.split('/')[-2]+'/'+root.find('path').text.split('/')[-1]
	        width, height = int(root.find('size').find('width').text), int(root.find('size').find('height').text)
	        
	        for obj in root.findall('object'):
	        
		        cls = obj.find('name').text
                bx = [float(obj.find('bndbox').find('xmin').text), float(obj.find('bndbox').find('xmax').text), float(obj.find('bndbox').find('ymin').text), float(obj.find('bndbox').find('ymax').text)]
                shapes.append((cls, bx))

            if(pattern.match(images.split('/')[-1]) and part=='eval'):
                self.add_image('bags', image_id = count, path = img_path, width=width, height=height, bags=shapes)
                
            if(not pattern.match(images.split('/')[-1]) and part=='train'):
                self.add_image('bags', image_id = count, path = img_path, width=width, height=height, bags=shapes)
            
            count+=1
        
        if (part == 'train'):
            for class_path in glob.glob('Data/bags/*'):
                for file_path in glob.glob(class_path):
                    self.add_image('bags', image_id = count, path = file_path, width=cv2.imread(file_path).shape[1], height=cv2.imread(file_path).shape[0], bags=[class_path.split('/')[-1], [0, cv2.imread(file_path).shape[1], 0, cv2.imread(file_path).shape[1]])
                    count+=1

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        path = info['path']
        return cv2.imread(path)[...,::-1]	    

    def image_reference(self, image_id):
        """Return the bags data of the image."""
        info = self.image_info[image_id]
        return info["bags"]

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['bags']
        
        masks = find_object_masks(info['path'], shapes)
        
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return masks, class_ids.astype(np.int32)


# In[5]:


# Training dataset
dataset_train = BagsDataset()
dataset_train.load_bags('train')
dataset_train.prepare()

# Validation dataset
dataset_val = BagsDataset()
dataset_val.load_shapes('eval')
dataset_val.prepare()

# ## Ceate Model

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# In[8]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')


# In[9]:


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")


# In[10]:


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


# ## Detection

# In[11]:


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# In[12]:


# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


# In[13]:


results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())


# ## Evaluation

# In[14]:


# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =        utils.compute_ap(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))

