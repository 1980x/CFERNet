'''
Aum Sri Sai Ram
Implementation of Compact Facial Expression Recognition Net for comparing against full lightcnn model
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
from lightcnn_full import LightCNN_29Layers_v2

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
    def __init__(self,num_classes=7, num_regions=4):
        super(FERNet, self).__init__()
        self.base= LightCNN_29Layers_v2(num_classes=7)
        
        checkpoint = torch.load('pretrained/LightCNN_29Layers_V2_checkpoint.pth.tar')
        pretrained_state_dict = dict(checkpoint['state_dict'])        
        keys = list(pretrained_state_dict.keys())        
        [pretrained_state_dict.pop(key) for key in keys if ( 'fc' in key)]   # for light cnn 29 v2    
        new_dict = dict(zip(list(self.base.state_dict().keys()), list(pretrained_state_dict.values())))
        self.base.load_state_dict(new_dict, strict = True)
        
        self.num_regions = num_regions
        
        self.eca = nn.ModuleList([eca_layer(128,3) for i in range(num_regions+1)])  
        
        self.globalavgpool = nn.ModuleList([nn.AdaptiveAvgPool2d(1)  for i in range(num_regions+1)])                                 
        self.region_net = nn.ModuleList([ nn.Sequential( nn.Linear(128,256), nn.ReLU()) for i in range(num_regions+1)])       
       
        self.classifiers =  nn.ModuleList([ nn.Linear(256+256, num_classes, bias = False) for i in range(num_regions+1)])
        self.s = 30.0
        
    def forward(self, x1, x2):
        x1 = self.base(x1)
        x2 = self.base(x2) 

        bs, c, w, h = x1.size()
        #print(x1.size())
        region_size = int(x1.size(2) / (self.num_regions/2) ) 
        
        patches1 = x1.unfold(2, region_size, region_size).unfold(3,region_size,region_size)         
        patches1 = patches1.contiguous().view(bs, c, -1, region_size, region_size).permute(0,2,1,3,4)
        patches2 = x2.unfold(2, region_size, region_size).unfold(3,region_size,region_size)         
        patches2 = patches2.contiguous().view(bs, c, -1, region_size, region_size).permute(0,2,1,3,4)         
                 
        output = []
        for i in range(int(self.num_regions)):
            f1 = patches1[:,i,:,:,:] 
            f1 = self.eca[i](f1) 
            f1 = self.globalavgpool[i](f1).squeeze(3).squeeze(2)
            f1 =  self.region_net[i](f1)
            
            f2 = patches2[:,i,:,:,:] 
            f2 = self.eca[i](f2) 
            f2 = self.globalavgpool[i](f2).squeeze(3).squeeze(2)
            f2 =  self.region_net[i](f2)
            
            f = torch.cat((f1,f2),dim=1) 
           
            f = self.s * self.classifiers[i](f)   
            output.append(f)      

        
        output_stacked = torch.stack(output, dim = 2)
        
       
        
        y1 = self.globalavgpool[4](self.eca[4](x1)).squeeze(3).squeeze(2)
        #y1 = self.globalavgpool[4](x1).squeeze(3).squeeze(2)     
        y1 = self.region_net[4](y1)
        
        y2 = self.globalavgpool[4](self.eca[4](x2)).squeeze(3).squeeze(2)
        #y2 = self.globalavgpool[4](x2).squeeze(3).squeeze(2)      
        y2 = self.region_net[4](y2)
          
        y = torch.cat((y1,y2),dim=1)                         
       
        y = torch.cat((y1,y2),dim=1)
        
        output_global = self.classifiers[4](y).unsqueeze(2)
        output_final = torch.cat((output_stacked,output_global),dim=2)
        
        return output_final
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        
        
if __name__=='__main__':
   model = FERNet() 
   model = model.to('cuda:0')
   print(count_parameters(model))
   x = torch.rand(2,  1, 128, 128).to('cuda:0')
   y = model(x, x) 
   print(y.size())
   
   #macs, params = profile(model, inputs=(x,x ))
   #macs, params = clever_format([macs, params], "%.3f")
   #print(macs,params)
   #print(y.size()) 
   '''
   for name, param in model.named_parameters():
       print(name, param.size())    
   '''    
