import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

class PruneLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PruneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        # Transpose to match the dimension
        self.mask = np.ones([self.out_features, self.in_features])
        m = self.in_features
        n = self.out_features
        self.sparsity = 0.0
        # Initailization
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

    def forward(self, x):
        out = self.linear(x)
        return out
        pass

    def prune_by_percentage(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        np_weight = self.linear.weight.data.cpu().numpy()
        flattened_weights = np.abs(np_weight.flatten())
        
        len_ = len(flattened_weights)
        random.seed(4)
        index = random.sample(range(0, len_), int(q/100*len_))
        #print(index)
        self.mask = np.ones(len_)
        self.mask[index] = 0
        self.mask = self.mask.reshape(np_weight.shape)
        self.mask = np.float32(self.mask)
        
        # Multiply weight by mask (Your code: 1 Line) 
        np_weight = np.multiply(np_weight, self.mask)
        # Copy back to linear.weight and assign to device (Your code: 1 Line)
        self.linear.weight.data = torch.from_numpy(np_weight).float().to(device)
        # Compute sparsity (Your code: 1 Line)
        self.sparsity = float((np_weight ==0).sum())/len(flattened_weights)
        # Copy mask to device for faster computation [Your code: 1 Line]
        self.mask = torch.from_numpy(self.mask).float().to(device)




class PrunedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(PrunedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # Expand and Transpose to match the dimension
        self.mask = np.ones([out_channels, in_channels, kernel_size, kernel_size])

        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0

    def forward(self, x):
        out = self.conv(x)
        return out

    def prune_by_percentage(self, q=5.0):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        np_weight = self.conv.weight.data.cpu().numpy()
        flattened_weights = np.abs(np_weight.flatten())
        
        len_ = len(flattened_weights)
        random.seed(4)
        index = random.sample(range(0, len_), int(q/100*len_))
        #print(index)
        self.mask = np.ones(len_)
        self.mask[index] = 0
        self.mask = self.mask.reshape(np_weight.shape)
        self.mask = np.float32(self.mask)
        
        # Multiply weight by mask (Your code: 1 Line) 
        np_weight = np.multiply(np_weight, self.mask)
        # Copy back to linear.weight and assign to device (Your code: 1 Line)
        self.conv.weight.data = torch.from_numpy(np_weight).float().to(device)
        # Compute sparsity (Your code: 1 Line)
        self.sparsity = float((np_weight ==0).sum())/len(flattened_weights)
    
        # Copy mask to device for faster computation [Your code: 1 Line]
        self.mask = torch.from_numpy(self.mask).float().to(device)




