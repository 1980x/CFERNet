"""
Aum Sri Sai Ram
Implementation of Customized MobileNet: It uses first few layers of MobileNet defined in mobilenetv2_archiecture.py and then local and global context branches are used along with ECA attention as shown in the Figure: Pipeline of CERN archiecture.
Ref:  Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. CVPR, pages 4510–4520, 2018
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-05-2021
Email: darshangera@sssihl.edu.in
Source: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

Customized Mobilenetv2 for ablation study in CERN


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

#from torchvision import models
from mobilenetv2_archiecture import mobilenet_v2  #This is trimmed : untill 96x14x14 layer
from thop import profile #uncomment this 
from thop import clever_format

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
       
    def __init__(self, in_channels=3, num_classes=7):
        super(FERNet, self).__init__()
        
        self.base = mobilenet_v2(pretrained=True)
        
        self.num_regions = 4
        
        self.eca = nn.ModuleList([eca_layer(192,3) for i in range( self.num_regions+1)])  
        
        self.globalavgpool = nn.ModuleList([nn.AdaptiveAvgPool2d(1)  for i in range( self.num_regions+1)])                                 
        self.region_net = nn.ModuleList([ nn.Sequential( nn.Linear(192,256), nn.ReLU()) for i in range( self.num_regions+1)])       
       
        self.classifiers =  nn.ModuleList([ nn.Linear(256, num_classes, bias = False) for i in range( self.num_regions+1)])
        self.s = 30.0
    def forward(self, x, y):
        x1 = self.base(x)
        #print(x1.size())
        bs, c, w, h = x1.size()
        region_size = int(x1.size(2) / (self.num_regions/2) ) 
        
        patches1 = x1.unfold(2, region_size, region_size).unfold(3,region_size,region_size)         
        patches1 = patches1.contiguous().view(bs, c, -1, region_size, region_size).permute(0,2,1,3,4)
        output = []
        for i in range(int(self.num_regions)):
            f1 = patches1[:,i,:,:,:] 
            f1 = self.eca[i](f1) 
            f1 = self.globalavgpool[i](f1).squeeze(3).squeeze(2)
            f1 =  self.region_net[i](f1)
                    
            f = f1 #torch.cat((f1,f2),dim=1) 
            
            for W in self.classifiers[i].parameters():
                W = F.normalize(W, p=2, dim=1)         
            f  = F.normalize(f, p=2, dim=1)
            
            f = self.s * self.classifiers[i](f)   
            output.append(f)      

        
        output_stacked = torch.stack(output, dim = 2)
        
        y1 = self.globalavgpool[4](self.eca[4](x1)).squeeze(3).squeeze(2)
        #y1 = self.globalavgpool[4](x1).squeeze(3).squeeze(2)     
        y1 = self.region_net[4](y1)
       
        
        for W in self.classifiers[4].parameters():
                W = F.normalize(W, p=2, dim=1)
                
                          
        y  = F.normalize(y1, p=2, dim=1)
        
        
        
        output_global = self.classifiers[4](y).unsqueeze(2)
        output_final = torch.cat((output_stacked,output_global),dim=2)
        
        return output_final
        
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        
        
if __name__=='__main__':
   model = FERNet() 
   
   print(count_parameters(model))
   x = torch.rand(1,3, 224, 224)
   y = model(x, x) 
   macs, params = profile(model, inputs=(x,x ))
   macs, params = clever_format([macs, params])
   print(macs,params)
   print(y.size()) 
          
        
        
        
        
        
        
        
        
        
        
        
        
        






