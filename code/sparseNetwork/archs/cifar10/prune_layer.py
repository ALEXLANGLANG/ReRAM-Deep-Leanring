import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

IB = 4
FB = 11
len_bits = 1+ IB+FB 
device = "cuda" if torch.cuda.is_available() else "cpu"

coeff = [0] 
for i in np.linspace(IB-1,-FB,IB+FB):
    coeff += [2**i]  
coeff = torch.tensor(coeff, dtype=torch.float32).to(device)

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

        #Inject SAF
        self.binary_ = None
        self.index  = None
        self.mask_bit_position = None
        self.index_mask = None #bit position to be masked
        self.set_bit = None
        self.mask_to_keep_SAF = None # mask to keep the weights SAF the same, in other words, don't update those weights
        
    def forward(self, x):
        out = self.linear(x)
        return out


    def prune_by_percentage(self, percentile_value=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """

        self.mask[torch.abs(self.linear.weight.data) < percentile_value ] = 0
        
        
    def set_up(self, mask_bit_position = [0]*16,q = 0.0,set_bit = 0,rs = 4):
        flattened_weights = torch.flatten(self.linear.weight.data)
        self.nonzero_indexes = torch.nonzero(flattened_weights).to(device)
        len_ = len(torch.flatten(self.nonzero_indexes))
        
        random.seed(rs)
        #index_: indexes of q percent weights within nonzero weights 
        index_ = torch.tensor(random.sample(range(0, len_), int(q/100*len_)),dtype=torch.int64).to(device) 
        #self.index: indexes of weights selected wthinin the whole flatten weights
        self.index = self.nonzero_indexes[index_].to(device)
        self.binary_ = torch.tensor([0]*len_bits*len(self.index)).to(device)
        self.mask_bit_position = torch.tensor(mask_bit_position).to(device)
        self.set_bit = set_bit
        self.index_mask = torch.where(self.mask_bit_position==1)[0].to(device)
        
        #Initialize the mask_to_keep_SAF
        mask_tmp = torch.ones_like(flattened_weights).to(device)
        mask_tmp[self.index] = 0
        self.mask_to_keep_SAF = mask_tmp.reshape(self.linear.weight.data.shape)
        
        
    def inject_SAF(self):
        #convert it into fixed point
        np_weight = self.linear.weight.data.clone().detach().to(device)
        unit = 2 ** -FB
        np_weight = torch.clamp(np_weight, max=2 ** IB - 2**(-FB), min=-(2 ** IB - 2**(-FB)))
        np_weight = torch.round(np_weight / unit) * unit
        
        #Do bit mask for each element of weight
        flattened_weights = torch.flatten(np_weight)
        mask_weights = flattened_weights[self.index]
        
        self.binary_ = self.binary_ * 0.0
        self.binary_ = torch.reshape(self.binary_,(len(self.index), len_bits)) 
        index_neg = torch.where(mask_weights<0)[0].to(device)
        mask_weights[index_neg] = - mask_weights[index_neg] #change negative into positive

        #convert x_fixed into binary representation
        mask_weights = torch.flatten(mask_weights*2**(FB))
        for i in range(len_bits -1):
            j = len_bits - i -1
            self.binary_[:,j] = torch.floor(mask_weights % 2).to(device)
            mask_weights = mask_weights//2

        self.binary_[:,0] = 0
        #Do masking and convert binary back to fixed point decimal
        self.binary_ = self.binary_.type(torch.float32)       
        self.binary_[:,self.index_mask] = self.set_bit
        mask_weights = torch.matmul(self.binary_,coeff)
        mask_weights[index_neg] = - mask_weights[index_neg]
        flattened_weights[self.index] = torch.reshape(mask_weights,flattened_weights[self.index].shape)                  
        self.linear.weight.data =  flattened_weights.reshape(np_weight.shape).to(device)






class PrunedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(PrunedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0
        
        self.mask = torch.ones_like(self.conv.weight.data).to(device)
        assert(self.mask.shape == self.conv.weight.data.shape)
        
        #Inject SAF
        self.binary_ = None
        self.index  = None
        self.mask_bit_position = None
        self.index_mask = None #bit position to be masked
        self.set_bit = None
        
        self.mask_to_keep_SAF = None # mask to keep the weights SAF the same, in other words, don't update those weights
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
        self.mask[torch.abs(self.conv.weight.data) < percentile_value ] = 0


    def set_up(self, mask_bit_position = [0]*16,q = 0.0,set_bit = 0,rs = 4):
        flattened_weights = torch.flatten(self.conv.weight.data)
        self.nonzero_indexes = torch.nonzero(flattened_weights).to(device)
        len_ = len(torch.flatten(self.nonzero_indexes))
        
        random.seed(rs)
        #index_: indexes of q percent weights within nonzero weights 
        index_ = torch.tensor(random.sample(range(0, len_), int(q/100*len_)),dtype=torch.int64).to(device) 
        #self.index: indexes of weights selected wthinin the whole flatten weights
        self.index = self.nonzero_indexes[index_].to(device)
        
        self.binary_ = torch.tensor([0]*len_bits*len(self.index)).to(device)
        self.mask_bit_position = torch.tensor(mask_bit_position).to(device)
        self.set_bit = set_bit
        self.index_mask = torch.where(self.mask_bit_position==1)[0].to(device)
        
        
        
        #Initialize the mask_to_keep_SAF
        mask_tmp = torch.ones_like(flattened_weights).to(device)
        mask_tmp[self.index] = 0
        self.mask_to_keep_SAF = mask_tmp.reshape(self.conv.weight.data.shape)
        
    def inject_SAF(self):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        #convert it into fixed point
        np_weight = self.conv.weight.data.clone().detach().to(device)
        unit = 2 ** -FB
        np_weight = torch.clamp(np_weight, max=2 ** IB - 2**(-FB), min=-(2 ** IB - 2**(-FB)))
        np_weight = torch.round(np_weight / unit) * unit
        
        #Do bit mask for each element of weight
        flattened_weights = torch.flatten(np_weight)
        mask_weights = flattened_weights[self.index]
        self.binary_ = self.binary_ * 0.0
        self.binary_ = torch.reshape(self.binary_,(len(self.index), len_bits)) 
        
        index_neg = torch.where(mask_weights<0)[0].to(device)
        mask_weights[index_neg] = - mask_weights[index_neg] #change negative into positive

        #convert x_fixed into binary representation
        mask_weights = torch.flatten(mask_weights*2**(FB))
        for i in range(len_bits -1):
            j = len_bits - i -1

            self.binary_[:,j] = torch.floor(mask_weights % 2).to(device)
            mask_weights = mask_weights//2

        self.binary_[:,0] = mask_weights*0
        
        #Do masking and convert binary back to fixed point decimal
        self.binary_ = self.binary_.type(torch.float32)       
        self.binary_[:,self.index_mask] = self.set_bit
        
        mask_weights =  torch.matmul(self.binary_,coeff)
        mask_weights[index_neg] = - mask_weights[index_neg]
       
        flattened_weights[self.index] = torch.reshape(mask_weights,flattened_weights[self.index].shape)               
        self.conv.weight.data =  flattened_weights.reshape(np_weight.shape).to(device)
