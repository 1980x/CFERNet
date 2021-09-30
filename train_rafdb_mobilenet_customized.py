'''
Aum Sri Sai Ram
Implementation of Customized MobileNet: This model is defined in mobilenet_branches.py. It uses first few layers of MobileNet and then local and global context branches are 
used along with ECA attention as shown in the Figure: Pipeline of CERN archiecture.
Ref:  Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. CVPR, pages 4510–4520, 2018
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 28-09-2021
Email: darshangera@sssihl.edu.in
'''
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np
import math
import argparse
import os
import time

from auto_augment import AutoAugment
from mobilenet_branches import FERNet as net, count_parameters
from dataset.sampler import ImbalancedDatasetSampler
from dataset.rafdb_dataset_mirror import ImageList

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if   torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Imponderous FER')

parser.add_argument('--root_path', type=str, default='../data/RAFDB/Image/aligned/',
                    help='path to root path of images')
parser.add_argument('--database', type=str, default='RAFDB',
                    help='Which Database for train. (RAFDB, Flatcam, FERPLUS)')
parser.add_argument('--train_list', type=str, default = '../data/RAFDB/EmoLabel/train_label.txt',
                    help='path to training list')
parser.add_argument('--test_list', type=str, default ='../data/RAFDB/EmoLabel/test_label.txt',  help='path to test list')
 
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',   help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr1', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr2', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,  metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',   help='path to latest checkpoint (default: checkpoints_affectnet/best_of_all_checkpoint.pth.tar)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--model_dir','-m', default='checkpoints_rafdb', type=str)
parser.add_argument('--imagesize', type=int, default = 224, help='image size (default: 224)')
parser.add_argument('--num_classes', type=int, default=7, help='number of expressions(class)')
parser.add_argument('--num_regions', type=int, default=4, help='number of non-overlapping patches(default:4)')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader:Resample, DRW,Reweight, None')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type:Focal, CE')
parser.add_argument('--confusion_plot', default=0, type=int, help='plot confusion plot or not 1/0')

best_prec1 = 0.


class_names = ['Neutral', 'Happy','Sad','Surprise', 'Fear', 'Disgust','Anger']# , 'Contempt']
def main():
    global args, best_prec1
    args = parser.parse_args()

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    imagesize = args.imagesize
    
    print(args)
      
    train_transform = transforms.Compose([                      
            AutoAugment(),
            transforms.Resize((args.imagesize,args.imagesize)),            
            transforms.ToTensor(),            
        ])
    
    valid_transform = transforms.Compose([
            transforms.Resize((args.imagesize,args.imagesize)),            
            transforms.ToTensor(),            
        ])
        
    train_dataset = ImageList(root=args.root_path, fileList=args.train_list,
                  transform=train_transform)
    cls_num_list = train_dataset.get_cls_num_list()    

    if args.train_rule == 'None':
       train_sampler = None  
       per_cls_weights = None 
    elif args.train_rule == 'Resample':
       train_sampler = ImbalancedDatasetSampler(train_dataset)
       per_cls_weights = None
    elif args.train_rule == 'Reweight':
       train_sampler = None
       beta = 0.9999                
       effective_num = 1.0 - np.power(beta, cls_num_list)
       per_cls_weights = (1.0 - beta) / np.array(effective_num)
       per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
       per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
    if args.loss_type == 'CE':
       criterion = nn.CrossEntropyLoss(weight=per_cls_weights).to(device)
    elif args.loss_type == 'Focal':
       criterion = FocalLoss(weight=per_cls_weights, gamma=2).to(device)
    else:
       warnings.warn('Loss type is not listed')
       return    
        
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)    
    
    test_data = ImageList(root=args.root_path, fileList=args.test_list,
                  transform=valid_transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, 
                         shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # prepare model
    basemodel = net(num_classes= args.num_classes)
    basemodel = torch.nn.DataParallel(basemodel).to(device)   
    print('\nNumber of trainable parameters: {}\n'.format(count_parameters(basemodel)))  
    
    params1 = {"params":[], "lr":args.lr1}   
    params2 = {"params":[], "lr":args.lr2}
    for name,param in basemodel.named_parameters():
        if 'region' in name or 'eca' in name or 'classifiers' in name:
            params2["params"].append(param)
        else:
            params1["params"].append(param)   
    optimizer = torch.optim.Adamax([params1,params2], betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    
    if args.pretrained:
       print("\n=> loading checkpoint '{}'".format(args.pretrained))
       checkpoint = torch.load(args.pretrained)
       basemodel.load_state_dict(checkpoint['base_state_dict'],strict=True)
        
        
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            basemodel.load_state_dict(checkpoint['base_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
    
    
    print('\nTraining starting:\n')
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch        
        train(train_loader, basemodel,  criterion, optimizer, epoch)
        prec1 = validate(test_loader, basemodel, criterion,  epoch, scheduler)
        print("Epoch: {}   Test Acc: {}".format(epoch, prec1))
        print("=================================================================\n")

        # remember best prec@1 and save checkpoint
        is_best = prec1.item() > best_prec1
        best_prec1 = max(prec1.to(device).item(), best_prec1)
        
        if is_best:
            print('So far best epoch, acc',epoch, best_prec1)
            torch.save(  {
                           'epoch': epoch + 1,
                           'base_state_dict': basemodel.state_dict(),
                           'best_prec1': best_prec1,
                           'optimizer' : optimizer.state_dict(),
                         }, os.path.join(args.model_dir, 'best_checkpoint_rafdb_mobilenet_customized.pth.tar'))
        
def train(train_loader,  basemodel, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    overall_loss = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    loss_ce = AverageMeter()
    loss_xent = AverageMeter()
    losses = AverageMeter()
    region_prec = []
     
    end = time.time()

    for i, (input1,input2, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input1 = input1.to(device)
        input2 = input2.to(device)
        target = target.to(device)
       
        # compute output
        region_preds = basemodel(input1, input2)
 
        #Region Branch Loss: loss2        
        for j in range(5):
            if j == 0:
               loss_ce = criterion(region_preds[:,:,j], target) #region celoss loss from Ist region branch 
            else:
               loss_ce += criterion(region_preds[:,:,j], target) #region celoss loss for rest 4 regions from region branch
       
        loss  = loss_ce 
        overall_loss.update(loss.item(), input1.size(0))
        avg_predictions = torch.mean(region_preds, dim=2)
        avg_prec = accuracy(avg_predictions,target,topk=(1,))       
        top1.update(avg_prec[0], input1.size(0))

        # compute gradient and update
        optimizer.zero_grad()        
        loss.backward()        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Training Epoch: [{0}][{1}/{2}]\t'
                  'BTime  ({batch_time.avg:.3f})\t'
                  'DTime ({data_time.avg:.3f})\t'
                  'overall_loss ({overall_loss.avg:.3f})\t' 
                  'Prec1  ({top1.avg:.3f}) \t'.format(
                   epoch, i, len(train_loader), data_time=data_time, batch_time=batch_time,                    
                  overall_loss=overall_loss,  top1=top1))

def validate(val_loader,  basemodel, criterion, epoch, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    overall_loss = AverageMeter()
    region_prec = []
    mode =  'Testing'

    # switch to evaluate mode
    basemodel.eval()
   
    end = time.time()

    with torch.no_grad():         
        for i, (input1,input2, target) in enumerate(val_loader):        
            data_time.update(time.time() - end)
            input1 = input1.to(device)
            input2 = input2.to(device)
            target = target.to(device)
            region_preds = basemodel(input1, input2)
            for j in range(5):
                if j == 0:
                   loss = criterion(region_preds[:,:,j], target) #region celoss loss from Ist global context branch 
                else:
                   loss += criterion(region_preds[:,:,j], target) #region celoss loss for rest 4 regions from local context branch
            
            overall_loss.update(loss.item(), input1.size(0))
     
            avg_predictions = torch.mean(region_preds, dim=2)
            avg_prec = accuracy(avg_predictions,target,topk=(1,))       
            top1.update(avg_prec[0], input1.size(0))

            top1.update(avg_prec[0], input1.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            #print( time.time() - end, batch_time.avg)
            end = time.time()
            
        print('\n{0} [{1}/{2}]\t'
                  'overall_loss ({overall_loss.avg})\t' 
                  'Prec@1  ({top1.avg})\t'
                  'Time@  ({batch_time.avg})\t'
                  .format(mode, i, len(val_loader), overall_loss=overall_loss,  top1=top1, batch_time=batch_time))

        scheduler.step(overall_loss.avg)
    return top1.avg
    

def validate_confusionplot(val_loader,  basemodel, criterion, epoch, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    overall_loss = AverageMeter()
    region_prec = []
    mode =  'Testing'

    # switch to evaluate mode
    basemodel.eval()
   
    end = time.time()

    with torch.no_grad():         
        for i, (input1,input2, target) in enumerate(val_loader):        
            data_time.update(time.time() - end)
            input1 = input1.to(device)
            input2 = input2.to(device)
            target = target.to(device)
            region_preds = basemodel(input1, input2)
            for j in range(5):
                if j == 0:
                   loss = criterion(region_preds[:,:,j], target) #region celoss loss from Ist region branch 
                else:
                   loss += criterion(region_preds[:,:,j], target) #region celoss loss for rest 3 regions from region branch
            
            overall_loss.update(loss.item(), input1.size(0))
            
            
     
            avg_predictions = torch.mean(region_preds, dim=2)
            _, avg_pred = torch.max(avg_predictions, 1)
            
            avg_prec = accuracy(avg_predictions,target,topk=(1,))
            if i == 0:
               all_predicted = avg_pred
               all_targets = target
            else:                   
               all_predicted = torch.cat((all_predicted, avg_pred), 0)
               all_targets = torch.cat((all_targets, target), 0)
            
                               

            top1.update(avg_prec[0], input1.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            
            end = time.time()
            
        print('\n{0} [{1}/{2}]\t'
                  'overall_loss ({overall_loss.avg})\t' 
                  'Prec@1  ({top1.avg})\t'
                  'Time@  ({batch_time.avg})\t'
                  .format(mode, i, len(val_loader), overall_loss=overall_loss,  top1=top1, batch_time=batch_time))

        scheduler.step(overall_loss.avg)
    return all_targets, all_predicted, top1.avg
    
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.rcParams.update({'font.size': 18}) 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
