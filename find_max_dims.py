import glob
import os
import numpy as np
import xml.etree.ElementTree as ET  

max_width, max_height = 0, 0

for f in glob.glob(os.getcwd()+'/Data/handbag_images/Annotations/*'):
    
    tree = ET.parse(f)
    root = tree.getroot()         
    width, height = int(root.find('size').find('width').text), int(root.find('size').find('height').text)
    
    if(width>max_width):
        max_width = width
    if(height>max_height):
        max_height = height 

print (max_height, max_width)       
