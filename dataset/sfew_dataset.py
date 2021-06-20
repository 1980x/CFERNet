"""
Aum Sri Sai Ram
By DG on 06-07-21

Dataset class for SFEW

Purpose: To return images from SFEW

Output:  bs x c x w x h        
            
"""

import torch.utils.data as data
from PIL import Image, ImageFile
import os
import pickle
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.utils import make_grid
ImageFile.LOAD_TRUNCATED_IAMGES = True
import random as rd 

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def default_reader(fileList, num_classes):
    imgList = []

    num_per_cls_dict = dict()
    for i in range(0, num_classes):
        num_per_cls_dict[i] = 0


    with open(fileList, 'r') as fp:
        for line in fp.readlines():  
            
            imgPath  = line.strip().split(' ')[0] #folder/imagename
            expression = int(line.strip().split(' ')[1])#emotion label
            imgList.append([imgPath, expression])
            num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1 
        fp.close()
        print('Total included ', len(imgList))
        return imgList,num_per_cls_dict


#Affectnet labels
def get_class(idx):
        classes = {
            0: 'Neutral',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Fear',
            5: 'Disgust',
            6: 'Anger',
            7: 'Contempt'}

        return classes[idx]





class ImageList(data.Dataset):
    def __init__(self, root, fileList,  transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = 7
        self.imgList, self.num_per_cls_dict = list_reader(fileList, self.cls_num)
        self.transform = transform
        self.loader = loader
        self.is_save = True
        self.totensor = transforms.ToTensor()
        


    def __getitem__(self, index):
        imgPath, target_expression = self.imgList[index]
        #print(imgPath, target_expression)
        img1 = self.loader(os.path.join(self.root, imgPath))
        
        img2 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1,img2, target_expression

    def __len__(self):
        return len(self.imgList)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

if __name__=='__main__':
   rootfolder= '../data/SFEW/Train_Aligned_Faces/'
   filename = '../data/SFEW/sfew_train.txt'


   transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor()])
   dataset = ImageList(rootfolder, filename, transform)

   fdi = iter(dataset)
   img_list = []
   target_list = []
   for i, data in enumerate(fdi):
       if i < 2:
          print(data[0][0].size(), data[1])
          continue
       else:
          break
   
