from coco import BagsConfig as C
import json
import numpy as np
import argparse

class ConfigSaver:

    def save(self, name):

        dic = {key:value for key, value in C.__dict__.items() if not key.startswith('__') and not callable(key)}
        
        new_dict = dic
        for key, val in dic.items():
            if type(val).__module__ == np.__name__:
                new_dict[key] = val.tolist()
                
        with open(name, 'w') as f:
            json.dump(new_dict, f)

if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--filename', help="Name of output config file", required=True)
    args = ap.parse_args()

    saver = ConfigSaver()
    saver.save(args.filename)
