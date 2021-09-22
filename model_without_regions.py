'''
Aum Sri Sai Ram
Implementation of Compact Facial Expression Recognition Net without local context branch (Only Global context branch)
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-05-2021
Email: darshangera@sssihl.edu.in
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

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
    def __init__(self,num_classes=7, num_regions=0):
        super(FERNet, self).__init__()
        self.base= LightCNN_29Layers_v2(num_classes=7)
        
        checkpoint = torch.load('pretrained/LightCNN_29Layers_V2_checkpoint.pth.tar')
        pretrained_state_dict = dict(checkpoint['state_dict'])        
        keys = list(pretrained_state_dict.keys())        
        [pretrained_state_dict.pop(key) for key in keys if ('3' in key or '4' in key or 'fc' in key)]   # for light cnn 29 v2    
        new_dict = dict(zip(list(self.base.state_dict().keys()), list(pretrained_state_dict.values())))
        self.base.load_state_dict(new_dict, strict = True)
                        
        self.eca = nn.ModuleList([eca_layer(192,3) for i in range(num_regions+1)])  
        
        self.globalavgpool = nn.ModuleList([nn.AdaptiveAvgPool2d(1)  for i in range(num_regions+1)])                                 
        self.region_net = nn.ModuleList([ nn.Sequential( nn.Linear(192,256), nn.ReLU()) for i in range(num_regions+1)])       
       
        self.classifiers =  nn.ModuleList([ nn.Linear(256+256, num_classes, bias = False) for i in range(num_regions+1)])
        self.s = 30.0
        
    def forward(self, x1, x2):
        x1 = self.base(x1)
        x2 = self.base(x2) 

        bs, c, w, h = x1.size()
        
        
        y1 = self.globalavgpool[0](self.eca[0](x1)).squeeze(3).squeeze(2)
             
        y1 = self.region_net[0](y1)
        
        y2 = self.globalavgpool[0](self.eca[0](x2)).squeeze(3).squeeze(2)
              
        y2 = self.region_net[0](y2)
        
        for W in self.classifiers[0].parameters():
                W = F.normalize(W, p=2, dim=1)
                
        y = torch.cat((y1,y2),dim=1)                         
        y  = F.normalize(y, p=2, dim=1)
        
        
        
        output_global = self.classifiers[0](y)
        
        
        return output_global
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        
        
if __name__=='__main__':
   model = FERNet() 
   model = model.to(device)
   print(count_parameters(model))
   x = torch.rand(2,  1, 128, 128).to(device)
   y = model(x, x) 
   print(y.size())
   '''
   for name, param in model.named_parameters():
       print(name, param.size())    
   '''    
