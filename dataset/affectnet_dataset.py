'''
Aum Sri Sai Ram
By Darshan on 08-05-20

Reference:
  Mollahosseini, A., Hasani, B. and Mahoor, M.H., 2017. AffectNet: A database for facial expression, valence,
    and arousal computing in the wild. IEEE Transactions on Affective Computing, 10(1), pp.18-31.

0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,
7: Contempt, 8: None, 9: Uncertain, 10: No-Face

NOTE: Removing 8: None, 9: Uncertain, 10: No-Face while training for FLATCAM.

classes = {
            0: 'Neutral',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Fear',
            5: 'Disgust',
            6: 'Anger',
            7: 'Contempt'}

No of samples in Manuall annoated set for each of the class are below:
0:74874
1:134415
2:25459
3:14090
4:6378
 5:3803
 6:24882
7:3750

'''


import torch.utils.data as data
from PIL import Image, ImageFile
import os
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import io
from torchvision import transforms
import random
ImageFile.LOAD_TRUNCATED_IAMGES = True

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def switch_expression(expression_argument):
    switcher = {
         0:'neutral',
         1:'Happiness',
          2: 'Sadness',
        3: 'Surprise',
4: 'Fear', 5: 'Disgust', 6: 'Anger',
7: 'Contempt', 8: 'None', 9: 'Uncertain', 10: 'No-Face'
    }
    return switcher.get(expression_argument, 0) #default neutral expression

def default_reader(fileList, num_classes):
    imgList = []
    if fileList.find('validation.csv')>-1: #hardcoded for Affectnet dataset
       start_index = 0
       max_samples = 150000
    else:
       start_index = 1
       max_samples = 2000000


    num_per_cls_dict = dict()
    for i in range(0, num_classes):
        num_per_cls_dict[i] = 0

    if num_classes == 7:
       exclude_list = [7, 8,9,10]
    else:
       exclude_list = [8,9,10]

    expression_0 = 0
    expression_1 = 0
    expression_2 = 0
    expression_3 = 0
    expression_4 = 0
    expression_5 = 0
    expression_6 = 0
    expression_7 = 0

    '''
    Below Ist two options for occlusion and pose case and 3rd one for general
    '''
    f = open('../data/Affectnetmetadata/validation.csv','r')
    lines = f.readlines()

    #random.shuffle(lines) #random shuffle to get random 5000 images

    if fileList.find('occlusion') > -1:
       fp = open(fileList,'r')
       for names in fp.readlines():
           _, target, image_path,_  = names.split('/') 
           image_path = image_path.strip()
           #print(target, image_path) 
           for line in lines:
               if line.find(image_path)>-1:
                  
                  imgPath  = line.strip().split(',')[0] #folder/imagename
                  (x,y,w,h)  = line.strip().split(',')[1:5]#bounding box coordinates
            
                  expression = int(line.strip().split(',')[6])  
                  #print(imgPath, expression)
                  if expression not in exclude_list: #Adding only list of first 8 expressions 
                     imgList.append([imgPath,(int(x),int(y),int(w),int(h)), expression]) 
                     num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1                     
       fp.close()
       return imgList, num_per_cls_dict 
    elif fileList.find('pose') > -1:
       fp = open(fileList,'r')
       for names in fp.readlines():
           target, image_path  = names.split('/')
           image_path = image_path.strip()  
           #print(target, image_path) 
           for line in lines:
               if line.find(image_path) > -1:                  
                  imgPath  = line.strip().split(',')[0] #folder/imagename
                  (x,y,w,h)  = line.strip().split(',')[1:5]#bounding box coordinates            
                  expression = int(line.strip().split(',')[6])  
                  #print(imgPath, expression)
                  if expression not in exclude_list: #Adding only list of first 8 expressions 
                     imgList.append([imgPath,(int(x),int(y),int(w),int(h)), expression])
                     num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1
        
       fp.close()
       return imgList, num_per_cls_dict 
    
             
    else:   #training or validation

        fp = open(fileList, 'r')
        for line in fp.readlines()[start_index:]:  #Ist line is header for automated labeled images
            
            imgPath  = line.strip().split(',')[0] #folder/imagename
            (x,y,w,h)  = line.strip().split(',')[1:5]#bounding box coordinates
            
            expression = int(line.strip().split(',')[6])#emotion label
            #print(imgPath, (x,y,w,h), expression)
            if expression == 0:
               expression_0 = expression_0 + 1            
               if expression_0 > max_samples:
                  continue
  
            if expression == 1:
               expression_1 = expression_1 + 1
               if expression_1 > max_samples:
                  continue  

            if expression == 2:
               expression_2 = expression_2 + 1
               if expression_2 > max_samples:
                  continue  

            if expression == 3:
               expression_3 = expression_3 + 1
               if expression_3 > max_samples:
                  continue  

            if expression == 4:
               expression_4 = expression_4 + 1
               if expression_4 > max_samples:
                  continue  

            if expression == 5:
               expression_5 = expression_5 + 1
               if expression_5 > max_samples:
                  continue  

            if expression == 6:
               expression_6 = expression_6 + 1
               if expression_6 > max_samples:
                  continue  

            if expression == 7:
               expression_7 = expression_7 + 1
               if expression_7 > max_samples:
                  continue  

            if expression not in exclude_list: #Adding only list of first 8 expressions 
               imgList.append([imgPath,(int(x),int(y),int(w),int(h)), expression])
               num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1 
        fp.close()
        #print('Total included ', len(imgList))
        return imgList,num_per_cls_dict


class ImageList(data.Dataset):
    def __init__(self, root, fileList, num_classes=7, transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = num_classes
        self.imgList, self.num_per_cls_dict =  list_reader(fileList, self.cls_num)
        self.transform = transform
        self.loader = loader
        self.fileList  = fileList

    def __getitem__(self, index):
        imgPath, (x,y,w,h), target_expression = self.imgList[index]
        #print(imgPath, (x,y,w,h), target_expression)
        area = (x,y,w,h)   
         
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


'''
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25

def show_dataset(dataset, n=6):
  img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                   for i in range(24)))#len(dataset))))
  print(img.shape)
  plt.imshow(img)
  plt.show()
  plt.axis('off')

'''
if __name__=='__main__':
   
   filelist = default_reader('../data/Affectnetmetadata/training.csv',7)

   
   rootfolder= '../data/AffectNetdataset/Manually_Annotated_Images/'
   #rootfolder= '../data/AffectNetdataset/Automatically_Annotated_Images/'
   #filename= '../data/Affectnetmetadata/automatically_annotated.csv'
   #filename = '../data/Affectnetmetadata/occlusion_affectnet_list.txt'
   filename = '../data/Affectnetmetadata/training.csv'

   #transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
   transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ColorJitter(hue=.05),
    #transforms.ColorJitter(saturation=.25),
    #transforms.ColorJitter(brightness=.25),
    #transforms.ColorJitter(contrast=.5),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(20, resample=Image.BILINEAR),
    ])

   dataset = ImageList(rootfolder, filename, transform)
   show_dataset(dataset)
   '''
   fdi = iter(dataset)
   img_list = []
   target_list = []
   for i, data in enumerate(fdi):
       if i < 2:
          print(data[0][0].size(), data[1])
          continue
       else:
          break
   '''


