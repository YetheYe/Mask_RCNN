import os

os.system('nohup python3 coco.py train --json_file /home/ye/diebold/class_detector/merged_data/merged.json --model /home/ye/diebold/class_detector/Mask_RCNN/models/mask_rcnn_bags_0017.h5 > nohup-4.out 2>&1')
os.system('mv logs/*/*.h5 models/temp1')
os.system('nohup python3 coco.py train --json_file /home/ye/diebold/class_detector/merged_data/merged.json --model /home/ye/diebold/class_detector/Mask_RCNN/models/mask_rcnn_bags_0017.h5 > nohup-5.out 2>&1')
