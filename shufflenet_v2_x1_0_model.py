'''
Aum Sri Sai Ram
Implementation of ShuffleNet V2
Ref:  Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. CVPR, pages 6848–6856, 2018.
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
        self.model = models.shufflenet_v2_x1_0(pretrained=True)
        #print(self.model)
        self.model.fc = nn.Linear(1024, num_classes)                            
        
        for p in self.model.parameters():
            p.requires_grad = True
        
    def forward(self, x, y):
        return self.model(x)  
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        
        
if __name__=='__main__':
   model = FERNet() 
   print(model)
   print(count_parameters(model))
   x = torch.rand(1,3, 224, 224)
   y = model(x, x) 
   macs, params = profile(model, inputs=(x,x ))
   macs, params = clever_format([macs, params], "%.3f")
   print(macs,params)
   print(y.size())        
        
        
        
        
        
        
        
        
        
        
        
        
        






