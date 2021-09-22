'''
Aum Sri Sai Ram
Implementation of Emotion Net
Ref: AMES LEE, Linda Wang, and Alexander Wong.  Emotionnet nano: Anefficient deep convolutional neural network design for real-time facial ex-pression recognition.Frontiers in Artificial Intelligence, 3:105, 2020.
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-05-2021
Email: darshangera@sssihl.edu.in
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from thop import profile 
from thop import clever_format

class FERNet(nn.Module):
       
    def __init__(self,in_channels = 3, num_classes=7):
        super(FERNet, self).__init__()
        self.layer1 = nn.Conv2d(3, 11, 3, padding = 1 )
        self.layer2 = nn.Conv2d(11, 9, 3, padding = 1 )
        self.layer3 = nn.Conv2d(9, 11, 3, padding = 1 )
        self.layer4 = nn.Conv2d(11, 8, 3, padding = 1 )
        self.layer5 = nn.Conv2d(8, 11, 3, padding = 1 )
        self.layer6 = nn.Conv2d(11, 7, 3, padding = 1 )
        self.layer7 = nn.Conv2d(7, 11, 3, padding = 1 )
        self.layer8 = nn.Conv2d(11, 27, 3, padding = 1, stride=2 )
        
        self.identity1 = nn.Conv2d(11, 27, 1, stride=2)
        
        #cNN Block 1
        self.layer1_c1 = nn.Conv2d(27, 27, 3, padding = 1 )
        self.layer2_c1 = nn.Conv2d(27, 19, 3, padding = 1 )
        self.layer3_c1 = nn.Conv2d(19, 27, 3, padding = 1 )
        self.layer4_c1 = nn.Conv2d(27, 26, 3, padding = 1 )
        self.layer5_c1 = nn.Conv2d(26, 27, 3, padding = 1 )
        self.layer6_c1 = nn.Conv2d(27, 36, 3, padding = 1, stride = 2 )
        
        self.identity2 = nn.Conv2d(27, 64, 1, stride=2)
        
        #cNN Block 2
        self.layer1_c2 = nn.Conv2d(36, 64, 3, padding = 1 )
        self.layer2_c2 = nn.Conv2d(64, 39, 3, padding = 1 )
        self.layer3_c2 = nn.Conv2d(39, 64, 3, padding = 1 )
        self.layer4_c2 = nn.Conv2d(64, 24, 3, padding = 1 )
        self.layer5_c2 = nn.Conv2d(24, 64, 3, padding = 1 )
        
        self.layer6_c2= nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(64,num_classes)
        
        
        
    def forward(self,x,z):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))
        
        x4 = F.relu(self.layer4(x1+x3))
        x5 = F.relu(self.layer5(x4))
        x6 = F.relu(self.layer6(x1+x3+x5))
        x7 = F.relu(self.layer7(x6))
        x8 = F.relu(self.layer8(x1+x5+x7))
        
        id1 = self.identity1(x1+x3+x5)
        
        x1_c1 = F.relu(self.layer1_c1(x8))
        x2_c1 = F.relu(self.layer2_c1(x1_c1+id1))
        x3_c1 = F.relu(self.layer3_c1(x2_c1))
        x4_c1 = F.relu(self.layer4_c1(x3_c1+x1_c1))
        x5_c1 = F.relu(self.layer5_c1(x4_c1))
        x6_c1 = F.relu(self.layer6_c1(x3_c1+x1_c1+x5_c1+id1))
        
        id2 = self.identity2(x3_c1+x5_c1+id1+x8)
        
        x1_c2 = F.relu(self.layer1_c2(x6_c1))
        x2_c2 = F.relu(self.layer2_c2(x1_c2+id2))
        x3_c2 = F.relu(self.layer3_c2(x2_c2))
        x4_c2 = F.relu(self.layer4_c2(x3_c2+x1_c2+id2))
        x5_c2 = F.relu(self.layer5_c2(x4_c2))
        x6_c2 = self.layer6_c2(x3_c2+x1_c2+x5_c2+id2).view(-1,64)
        
        y = self.fc(x6_c2)
        
        return y
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        
        
if __name__=='__main__':
   model = FERNet() 
   model = model
   print(count_parameters(model))
   x = torch.rand(1,3, 48, 48)
   macs, params = profile(model, inputs=(x,x ))
   macs, params = clever_format([macs, params], "%.3f")
   print(macs,params)
   y = model(x,x) 
   print(y.size())        
        
        
        
        
        
        
        
        
        
        
        
        
        






                  