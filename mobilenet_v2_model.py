'''
Aum Sri Sai Ram
Implementation of MobilenetV2
Ref:  Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. CVPR, pages 4510–4520, 2018
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-05-2021
Email: darshangera@sssihl.edu.in
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torchvision import models
from thop import profile #uncomment this 
from thop import clever_format
class FERNet(nn.Module):
       
    def __init__(self, in_channels=3, num_classes=7):
        super(FERNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        #print(self.model.classifier)
        #print(self.model.classifier[1])
        """self.model.features[0] = nn.Sequential(
                                                nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), 
                                                nn.BatchNorm2d(32),
                                                nn.ReLU6(inplace=True)
                                               )"""

        
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                                              nn.Linear(1280, num_classes)
                                             )
        
        for p in self.model.parameters():
            p.requires_grad = True
        
    def forward(self, x, y):
        return self.model(x)  
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        
        
if __name__=='__main__':
   model = FERNet() 
  
   print(count_parameters(model))
   x = torch.rand(1,3, 224, 224)
   y = model(x, x) 
   macs, params = profile(model, inputs=(x,x ))
   macs, params = clever_format([macs, params], "%.3f")
   print(macs,params)
   print(y.size()) 
          
        
        
        
        
        
        
        
        
        
        
        
        
        






