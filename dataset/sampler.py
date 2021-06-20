
import torch
import torch.utils.data
import torchvision
import pdb
import numpy as np
#from  dataset.affectnet_dataset import ImageList as  affectnetImageList
from  dataset.affectnet_dataset_mirror import ImageList as  affectnetImageList
#from  dataset.rafdb_dataset import ImageList as rafdbImageList
from  dataset.rafdb_dataset_mirror import ImageList as rafdbImageList
#from  dataset.rafdb_affectnet_combined_mirror import ImageList as rafdbImageList

from  dataset.ferplus_dataset_mirror import ImageList as ferplusImageList

from dataset.noisy_affectnet_dataset_mirror import ImageList as  noisy_affectnetImageList

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        #print(self.indices)    
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        #print(self.num_samples)              
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            #print(label)
            # spdb.set_trace()
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        #print(dataset_type)
        #pdb.set_trace()
        if dataset_type is affectnetImageList:
            return dataset.imgList[idx][2]
        elif dataset_type is noisy_affectnetImageList:
            return dataset.imgList[idx][1]
        
        elif dataset_type is rafdbImageList:
            return dataset.imgList[idx][1]
        elif dataset_type is ferplusImageList:
            target = dataset.imgList[idx][1] 
            target = np.argmax(target) #majority class only handled
            return target  
        
     
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
