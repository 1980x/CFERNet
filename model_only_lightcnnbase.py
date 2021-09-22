'''
Aum Sri Sai Ram
Implementation of Compact Facial Expression Recognition Net without using local and global context branches
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-05-2021
Email: darshangera@sssihl.edu.in
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

from thop import profile
from thop import clever_format
from light_cnn import LightCNN_29Layers_v2

class eca_layer(nn.Module):
   
    def __init__(self, channel, k_size = 5):
        super(eca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
     
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class FERNet(nn.Module):
    def __init__(self,num_classes=7):
        super(FERNet, self).__init__()
        self.base= LightCNN_29Layers_v2(num_classes=7)
        
        checkpoint = torch.load('pretrained/LightCNN_29Layers_V2_checkpoint.pth.tar')
        pretrained_state_dict = dict(checkpoint['state_dict'])        
        keys = list(pretrained_state_dict.keys())        
        [pretrained_state_dict.pop(key) for key in keys if ('3' in key or '4' in key or 'fc' in key)]   # for light cnn 29 v2    
        new_dict = dict(zip(list(self.base.state_dict().keys()), list(pretrained_state_dict.values())))
        self.base.load_state_dict(new_dict, strict = True)
            
        self.globalavgpool = nn.AdaptiveAvgPool2d(1)
        self.classifiers = nn.Linear(192, num_classes, bias = False) 
    def forward(self, x1, x2):
        x1 = self.base(x1)
        f = self.globalavgpool(x1).squeeze(3).squeeze(2)        
        y = self.classifiers(f)   
        return y      

        
        
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        
        
if __name__=='__main__':
   model = FERNet() 
   model = model.to('cuda:0')
   print(count_parameters(model))
   x = torch.rand(1,  1, 128, 128).to('cuda:0')
   y = model(x, x) 
   print(y.size())
   macs, params = profile(model, inputs=(x,x ))
   macs, params = clever_format([macs, params], "%.3f")
   print(macs,params)
   print(y.size()) 
   '''
   for name, param in model.named_parameters():
       print(name, param.size())    
   '''    
