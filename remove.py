import glob
import os

for filename in glob.glob('Data/handbag_images/JPEGImages/*.npy'):
    os.remove(filename)

for filename in glob.glob('Data/bags/*'):
    for actual in glob.glob(filename+'/*.npy'):
        os.remove(actual)
        
