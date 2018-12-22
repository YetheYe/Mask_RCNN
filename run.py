import os
import shutil

shutil.copy('train.txt', 'store/train_old.txt')
shutil.copy('val.txt', 'store/val_old.txt')
shutil.copy('test.txt', 'store/test_old.txt')

os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 coco.py train --json_file /home/ye/diebold/class_detector/complete_data/labels.json --model /home/ye/diebold/class_detector/Mask_RCNN/models/mask_rcnn_coco.h5 --num_gpu 4  --lr 1e-3 --stage 3 > fins33.out 2>&1')

shutil.copy('train.txt', 'store/train_33.txt')
shutil.copy('val.txt', 'store/val_33.txt')
shutil.copy('test.txt', 'store/test_33.txt')

os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 coco.py train --json_file /home/ye/diebold/class_detector/complete_data/labels.json --model /home/ye/diebold/class_detector/Mask_RCNN/models/mask_rcnn_coco.h5 --num_gpu 4  --lr 1e-4 --stage 3 > fins34.out 2>&1')

shutil.copy('train.txt', 'store/train_34.txt')
shutil.copy('val.txt', 'store/val_34.txt')
shutil.copy('test.txt', 'store/test_34.txt')


os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 coco.py train --json_file /home/ye/diebold/class_detector/complete_data/labels.json --model /home/ye/diebold/class_detector/Mask_RCNN/models/mask_rcnn_coco.h5 --num_gpu 4  --lr 1e-5 --stage 3 > fins35.out 2>&1')

shutil.copy('train.txt', 'store/train_35.txt')
shutil.copy('val.txt', 'store/val_35.txt')
shutil.copy('test.txt', 'store/test_35.txt')

os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 coco.py train --json_file /home/ye/diebold/class_detector/complete_data/labels.json --model /home/ye/diebold/class_detector/Mask_RCNN/models/mask_rcnn_coco.h5 --num_gpu 4  --lr 1e-3 --stage 2 > fins23.out 2>&1')

shutil.copy('train.txt', 'store/train_23.txt')
shutil.copy('val.txt', 'store/val_23.txt')
shutil.copy('test.txt', 'store/test_23.txt')


os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 coco.py train --json_file /home/ye/diebold/class_detector/complete_data/labels.json --model /home/ye/diebold/class_detector/Mask_RCNN/models/mask_rcnn_coco.h5 --num_gpu 4  --lr 1e-4 --stage 2 > fins24.out 2>&1')
shutil.copy('train.txt', 'store/train_24.txt')
shutil.copy('val.txt', 'store/val_24.txt')
shutil.copy('test.txt', 'store/test_24.txt')


os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 coco.py train --json_file /home/ye/diebold/class_detector/complete_data/labels.json --model /home/ye/diebold/class_detector/Mask_RCNN/models/mask_rcnn_coco.h5 --num_gpu 4  --lr 1e-5 --stage 2 > fins25.out 2>&1')
shutil.copy('train.txt', 'store/train_25.txt')
shutil.copy('val.txt', 'store/val_25.txt')
shutil.copy('test.txt', 'store/test_25.txt')


os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 coco.py train --json_file /home/ye/diebold/class_detector/complete_data/labels.json --model /home/ye/diebold/class_detector/Mask_RCNN/models/mask_rcnn_coco.h5 --num_gpu 4  --lr 1e-3 --stage 1 > fins13.out 2>&1')
shutil.copy('train.txt', 'store/train_13.txt')
shutil.copy('val.txt', 'store/val_13.txt')
shutil.copy('test.txt', 'store/test_13.txt')


os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 coco.py train --json_file /home/ye/diebold/class_detector/complete_data/labels.json --model /home/ye/diebold/class_detector/Mask_RCNN/models/mask_rcnn_coco.h5 --num_gpu 4  --lr 1e-4 --stage 1 > fins14.out 2>&1')
shutil.copy('train.txt', 'store/train_14.txt')
shutil.copy('val.txt', 'store/val_14.txt')
shutil.copy('test.txt', 'store/test_14.txt')


os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 coco.py train --json_file /home/ye/diebold/class_detector/complete_data/labels.json --model /home/ye/diebold/class_detector/Mask_RCNN/models/mask_rcnn_coco.h5 --num_gpu 4  --lr 1e-5 --stage 1 > fins15.out 2>&1')
shutil.copy('train.txt', 'store/train_15.txt')
shutil.copy('val.txt', 'store/val_15.txt')
shutil.copy('test.txt', 'store/test_15.txt')


