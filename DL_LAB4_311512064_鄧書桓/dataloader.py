import pandas as pd
from torch.utils import data
import numpy as np
import torch
import os
import io
import torchvision.transforms as transforms
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode, transform = False):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode

        # set the transform
        self.transform = transform

        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        if torch.is_tensor(index):
            index = index.tolist()

        path = os.path.join(self.root, self.img_name[index] + '.jpeg')
        img = Image.open(path)
        
        min_size = img.size[0] if img.size[0]<img.size[1] else img.size[1] 

        if self.transform :
            transforms_pre = transforms.Compose([transforms.CenterCrop(min_size), 
                                             transforms.Resize((512, 512)),
                                             transforms.RandomHorizontalFlip(p = 0.5),
                                             transforms.RandomVerticalFlip(p = 0.5),
                                             transforms.RandomRotation(degrees = 10), 
                                             transforms.ToTensor()])
            img = transforms_pre(img)
        # vutils.save_image(img, './test/new_example.jpg')
        else:
            transforms_pre = transforms.Compose([transforms.CenterCrop(min_size), 
                                             transforms.Resize((512, 512)),
                                             transforms.ToTensor()])
            img = transforms_pre(img)

        label = self.label[index]
        return img, label
