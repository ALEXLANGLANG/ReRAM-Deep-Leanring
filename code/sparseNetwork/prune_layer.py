import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-6
class PruneLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PruneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        # Transpose to match the dimension
        m = self.in_features
        n = self.out_features
        self.sparsity = 0.0
        # Initailization
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))
        self.mask = torch.ones_like(self.linear.weight.data).to(device)
        assert(self.mask.shape == self.linear.weight.data.shape)
        
    def forward(self, x):
        out = self.linear(x)
        return out


    def prune_by_percentage(self, percentile_value=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
#         tensor = self.linear.weight.data.cpu().numpy()
#         alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
#         percentile_value = np.percentile(abs(alive), q)

        self.mask[torch.abs(self.linear.weight.data) < percentile_value ] = 0
        array_all_weights = self.linear.weight.data.cpu().numpy()
        q = 200/len(array_all_weights.flatten())*100
        percentile_value_remain = np.percentile(abs(np.array(array_all_weights)), 100. - 0.01)
        self.mask[torch.abs(self.linear.weight.data) > percentile_value_remain ] = 1




class PrunedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1, padding=0,groups =1,  bias=True):
        super(PrunedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups = groups, bias=bias)
        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0
        
        self.mask = torch.ones_like(self.conv.weight.data).to(device)
        assert(self.mask.shape == self.conv.weight.data.shape)

    def forward(self, x):
        out = self.conv(x)
        return out

    def prune_by_percentage(self, percentile_value=5.0):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
#         tensor = self.conv.weight.data.cpu().numpy()
#         alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
#         percentile_value = np.percentile(abs(alive), q)
#         percentile_value = q
        self.mask[torch.abs(self.conv.weight.data) < percentile_value ] = 0

        array_all_weights = self.conv.weight.data.cpu().numpy()
        q = 200/len(array_all_weights.flatten())*100
        percentile_value_remain = np.percentile(abs(np.array(array_all_weights)), 100. - 0.8) 
        self.mask[torch.abs(self.conv.weight.data) > percentile_value_remain ] = 1







