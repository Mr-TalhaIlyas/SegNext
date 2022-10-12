from cv2 import INTER_NEAREST
import yaml

with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch.utils.data as data
from fmutils import fmutils as fmu
from tabulate import tabulate
# from PIL import Image
import cv2
import numpy as np
import os, random, time

from augmenters import data_augmenter
from data_utils import std_norm, encode_labels

class GEN_DATA_LISTS():
    
    def __init__(self, root_dir, sub_dirname):
        '''
        Parameters
        ----------
        root_dir : TYPE
            root directory containing [train, test, val] folders.
        sub_dirname : TYPE
            sub directories inside the main split (train, test, val) folders
        get_lables_from : TYPE
            where to get the label from either from dir_name of file_name.


        '''
        self.root_dir = root_dir
        self.sub_dirname = sub_dirname
        self.splits = ['train', 'val', 'test']
        
    def get_splits(self):
        
        print('Directories loadded:')
        self.split_files = []
        for split in self.splits:
            print(os.path.join(self.root_dir, split, self.sub_dirname[0]))
            self.split_files.append(os.path.join(self.root_dir, split, self.sub_dirname[0]))
        print('\n')
        self.split_lbls = []
        for split in self.splits:
            print(os.path.join(self.root_dir, split, self.sub_dirname[1]))
            self.split_lbls.append(os.path.join(self.root_dir, split, self.sub_dirname[1]))
            
        
        self.train_f = fmu.get_all_files(self.split_files[0])
        self.val_f = fmu.get_all_files(self.split_files[1])
        self.test_f = fmu.get_all_files(self.split_files[2])

        self.train_l = fmu.get_all_files(self.split_lbls[0])
        self.val_l = fmu.get_all_files(self.split_lbls[1])
        self.test_l = fmu.get_all_files(self.split_lbls[2])
        
        train, val, test = [self.train_f, self.train_l], [self.val_f, self.val_l], [self.test_f, self.test_l]
        
        return train, val, test
    
    def get_classes(self):
        
        cls_names = []
        for i in range(len(self.train_f)):
            cls_names.append(fmu.get_basename(self.train_f[i]).split('_')[1])
        classes = sorted(list(set(cls_names)), key=fmu.numericalSort)
        return classes
    
    def get_filecounts(self):
        print('\n')
        result = np.concatenate((np.asarray(['train', 'val', 'test']).reshape(-1,1),
                                np.asarray([len(self.train_f), len(self.val_f), len(self.test_f)]).reshape(-1,1),
                                np.asarray([len(self.train_l), len(self.val_l), len(self.test_l)]).reshape(-1,1))
         , 1)
        print(tabulate(np.ndarray.tolist(result), headers = ["Split", "Images", "Labels"], tablefmt="github"))
        return None

class Cityscape(data.Dataset):
    def __init__(self, img_paths, mask_paths, img_height, img_width, augment_data=False, normalize=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_height = img_height
        self.img_width = img_width
        self.augment_data = augment_data
        self.normalize = normalize
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        # random.seed(time.time())
        data_sample = {}
        
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.img_width, self.img_height), cv2.INTER_LINEAR)

        lbl = cv2.imread(self.mask_paths[index], 0)
        # lbl = cv2.resize(lbl, (self.img_width, self.img_height), cv2.INTER_NEAREST)
        lbl = encode_labels(lbl)

        if self.augment_data:
            img,  lbl = data_augmenter(img, lbl)
            
        if self.normalize:
            img = std_norm(img)
        
        data_sample['img'] = img
        data_sample['lbl'] = lbl

        return data_sample 