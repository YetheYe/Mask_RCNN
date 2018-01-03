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
    
    def load_bags(self, part = 'train'):
        """Generate the requested number of synthetic images.
        part: train/eval
        """

        count = 0

        # Add classes
        
        classes = ['black_backpack', 'nine_west_bag', 'meixuan_brown_handbag', 'sm_bdrew_grey_handbag', 'wine_red_handbag', 'sm_bclarre_blush_crossbody', 'mk_brown_wrislet', 'black_plain_bag', 'lmk_brown_messenger_bag', 'sm_peach_backpack', 'black_ameligalanti', 'white_bag']
        
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
            
            for class_path in glob.glob('Data/bags/*'):
                for file_path in glob.glob(class_path):
                        
                        shapes = []
                        img = cv2.imread(file_path)[...,::-1]
                        height, width, _ = img.shape
            count+=1

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

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
        
        masks = 
        
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

