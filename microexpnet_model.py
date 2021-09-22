'''
Aum Sri Sai Ram
Implementation of MicroexpNet : Ilke Cugu, Eren Sener, and Emre Akbas. Microexpnet: An extremely small and fast model for expression recognition from face images. 
In 2019 Ninth International Conference on Image Processing Theory, Tools and Applications (IPTA), pages 1–6. IEEE, 2019
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-05-2021
Email: darshangera@sssihl.edu.in
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.transforms as transforms

from thop import profile #uncomment this 
from thop import clever_format

class FERNet(nn.Module):
       
    def __init__(self,in_channels = 1, num_classes=8):
        super(FERNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, stride=2, padding=6)
        
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        
        self.mp1 = nn.MaxPool2d(2, padding = 1)
        self.mp2 = nn.MaxPool2d(2, padding = 1)  
              
        self.fc1 = nn.Linear(32*6*6, 48)
        #self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(48, num_classes)
        
        for name,param in self.named_parameters():       
            #print(name,param)
            if 'weight' in name:     
               nn.init.xavier_normal_(param)
            elif 'bias' in name:
               param = torch.randn(param.size(0)) * 0.5 
               
        self.transform = transforms.Compose([        
                                  transforms.Resize((84,84)),
                                  transforms.Grayscale(),                                     
                                  ])                                 
                
        
    def forward(self,x,z):
        x = self.transform(x)
        x1 = self.mp1( F.relu(self.conv1(x)))
        x2 = self.mp2(F.relu(self.conv2(x1)))        
        x3 = F.relu(self.fc1(x2.view(x.size(0),-1)))
        y = self.fc2(x3)        
        return y
        
         
def kd_loss(teacherlogits, studentlogits, labels, T = 8, lambda_= 0.5):
    with torch.no_grad():
        outputTeacher = (1.0 / T) * teacherlogits 
        outputTeacher = F.softmax(outputTeacher, dim =1)
    cost_1 = F.cross_entropy(studentlogits, labels)
    pred = F.softmax(studentlogits, dim = 1)
    logp = F.log_softmax(studentlogits/T, dim=1)
    cost_2 = -torch.mean(torch.sum(outputTeacher * logp, dim=1))
    cost = ((1.0 - lambda_) * cost_1 + lambda_ * cost_2)
    return cost    
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        
        
if __name__=='__main__':
   model = FERNet() 
   x = torch.rand(1,3, 84,84)
   macs, params = profile(model, inputs=(x,x ))
   macs, params = clever_format([macs, params], "%.3f")
   print(macs,params)
   print(count_parameters(model))
  
   y = model(x,x) 
   print(y.size())        
        
                