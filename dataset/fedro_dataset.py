"""
Aum Sri Sai Ram

Modified on :07-07-2020

Test on FEDRO dataset provided by
Reference:
https://github.com/mysee1989/PG-CNN
Paper: Occlusion Aware Facial Expression Recognition Using CNN With Attention Mechanism

File: '../data/FED_RO/occlusion_emotion_dataset.txt'


"""


import torch.utils.data as data
from PIL import Image, ImageFile
import os
import sys
import csv
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

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)


def default_reader(fileList):

       counter_loaded_images_per_label = [0 for _ in range(8)]


       imgList = []

    #if fileList.find('occlusion_emotion_') > -1:
       fp = open(fileList,'r')
       for names in fp.readlines():
           target, image_path  = names.split('/')  #Eg. for each entry before underscore lable and afterwards name in 1_fer0034656.png 8 0, 2_fer0033878.png 8 0

           image_path = names.strip()#imagename is emotion/name.jpg
           '''
           if not os.path.exists('../data/FED_RO/FED_RO_aligned/'+image_path):
              print(image_path)
              continue
           '''
           target = map_emotion_label(target)

           imgList.append((image_path, int(target)))

       return imgList 
             
def map_emotion_label(emotion): #makes it same that of affectnet
    classes = {
            'neural':0,
            'disgust':5,
            'fear':4,
            'anger':6,    
            'happy':1,
            'surprise':3,
            'sad': 2,  }
    return classes[emotion]
'''
def get_class(idx):  #class expression label
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
'''
'''
def change_emotion_label_same_as_affectnet(emo_to_return):
        """
        Parse labels to make them compatible with AffectNet.  
        #https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/model/utils/udata.py
        """

        if emo_to_return == 2:
            emo_to_return = 3
        elif emo_to_return == 3:
            emo_to_return = 2
        elif emo_to_return == 4:
            emo_to_return = 6
        elif emo_to_return == 6:
            emo_to_return = 4

        return emo_to_return
'''
class ImageList(data.Dataset):
    def __init__(self, root, fileList,   transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader
       
        self.totensor = transforms.ToTensor()

       
    def __getitem__(self, index):
        imgPath, target_expression = self.imgList[index]
        img1 = self.loader(os.path.join(self.root, imgPath))

        img2 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1,img2, target_expression


    def __len__(self):
        return len(self.imgList)
    
    



if __name__=='__main__':
   testlist = default_reader('../data/FED_RO/occlusion_emotion_fedro_our_list.txt')
   for i in range(2):
       print(testlist[i])
   imagesize =  224
   transform = transforms.Compose([transforms.Resize((imagesize,imagesize)), transforms.ToTensor()])

   dataset = ImageList(root='../data/FED_RO/FED_RO_aligned/', fileList ='../data/FED_RO/occlusion_emotion_fedro_our_list.txt',transform=transform      )

   fdi = iter(dataset)
   for i, data in enumerate(fdi):
        if i < 2:
           print(' ',data[0].size(),data[1])
        else:
           break
   
