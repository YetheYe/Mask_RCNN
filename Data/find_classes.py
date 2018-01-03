import xml.etree.ElementTree as ET
import glob
import numpy as np
import cv2

classes, count = [], []

for filename in glob.glob('handbag_images/Annotations/*.xml'):
    
    tree = ET.parse(filename)
    root = tree.getroot()
    
    path = 'handbag_images/'+root.find('path').text.split('/')[-2]+'/'+root.find('path').text.split('/')[-1]
    width, height = int(root.find('size').find('width').text), int(root.find('size').find('height').text)
    
    #print (path, width, '*', height)
    
    for obj in root.findall('object'):
        
        if obj.find('name').text not in classes:
            classes.append(obj.find('name').text)
        if obj.find('name').text=='mk_brown_wrislet':
            img = cv2.imread(path)
            cv2.imshow('frame', img)
            cv2.waitKey(0)
            
        
    #print('......................................................')

print (classes)
