import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
import time

IB = 4
FB = 11
len_bits = 1 + IB + FB
device = "cuda" if torch.cuda.is_available() else "cpu"

coeff = [0]
for i in np.linspace(IB - 1, -FB, IB + FB):
    coeff += [2 ** i]
coeff = torch.tensor(coeff, dtype=torch.float32).to(device)


class PruneLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PruneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        m = self.in_features
        n = self.out_features
        # Initialization
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m + n)))
        self.binary_ = None
        self.index = None
        self.mask_bit_position = None
        self.index_mask = None  # bit position to be masked
        self.set_bit = None

    def forward(self, x):
        out = self.linear(x)
        return out
        pass

    def set_up(self, mask_bit_position=[0] * 16, q=0.0, set_bit=0, rs=4):
        flattened_weights = torch.flatten(self.linear.weight.data)
        len_ = len(flattened_weights)
        random.seed(rs)
        self.index = torch.tensor(random.sample(range(0, len_), int(q / 100 * len_)), dtype=torch.int64).to(device)
        self.binary_ = torch.tensor([0] * len_bits * len(self.index)).to(device)
        self.mask_bit_position = torch.tensor(mask_bit_position).to(device)
        self.set_bit = set_bit
        self.index_mask = torch.where(self.mask_bit_position == 1)[0].to(device)

    def prune_by_percentage(self):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        # convert it into fixed point
        np_weight = self.linear.weight.data.clone().detach().to(device)
        unit = 2 ** -FB
        np_weight = torch.clamp(np_weight, max=2 ** IB - 2 ** (-FB), min=-(2 ** IB - 2 ** (-FB)))
        np_weight = torch.round(np_weight / unit) * unit

        # Do bit mask for each element of weight
        flattened_weights = torch.flatten(np_weight)
        mask_weights = flattened_weights[self.index]
        self.binary_ = self.binary_ * 0.0
        self.binary_ = torch.reshape(self.binary_, (len(self.index), len_bits))
        index_neg = torch.where(mask_weights < 0)[0].to(device)
        mask_weights[index_neg] = - mask_weights[index_neg]  # change negative into positive

        # convert x_fixed into binary representation
        mask_weights = mask_weights * 2 ** (FB)
        for i in range(len_bits - 1):
            j = len_bits - i - 1
            self.binary_[:, j] = torch.floor(mask_weights % 2).to(device)
            mask_weights = mask_weights // 2

        self.binary_[:, 0] = 0

        # Do masking and convert binary back to fixed point decimal
        self.binary_ = self.binary_.type(torch.float32)
        self.binary_[:, self.index_mask] = self.set_bit
        mask_weights = torch.matmul(self.binary_, coeff)
        mask_weights[index_neg] = - mask_weights[index_neg]
        flattened_weights[self.index] = mask_weights
        self.linear.weight.data = flattened_weights.reshape(np_weight.shape).to(device)


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
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n + m)))
        self.binary_ = None
        self.index = None
        self.mask_bit_position = None
        self.index_mask = None  # bit position to be masked
        self.set_bit = None

    def forward(self, x):
        out = self.conv(x)
        return out

    def set_up(self, mask_bit_position=[0] * 16, q=0.0, set_bit=0, rs=4):
        flattened_weights = torch.flatten(self.conv.weight.data)
        len_ = len(flattened_weights)
        random.seed(rs)
        self.index = torch.tensor(random.sample(range(0, len_), int(q / 100 * len_)), dtype=torch.int64).to(device)
        self.binary_ = torch.tensor([0] * len_bits * len(self.index)).to(device)
        self.mask_bit_position = torch.tensor(mask_bit_position).to(device)
        self.set_bit = set_bit
        self.index_mask = torch.where(self.mask_bit_position == 1)[0].to(device)

    def prune_by_percentage(self):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        # convert it into fixed point
        np_weight = self.conv.weight.data.clone().detach().to(device)
        unit = 2 ** -FB
        np_weight = torch.clamp(np_weight, max=2 ** IB - 2 ** (-FB), min=-(2 ** IB - 2 ** (-FB)))
        np_weight = torch.round(np_weight / unit) * unit

        # Do bit mask for each element of weight
        flattened_weights = torch.flatten(np_weight)
        mask_weights = flattened_weights[self.index]
        self.binary_ = self.binary_ * 0.0
        self.binary_ = torch.reshape(self.binary_, (len(self.index), len_bits))
        index_neg = torch.where(mask_weights < 0)[0].to(device)
        mask_weights[index_neg] = - mask_weights[index_neg]  # change negative into positive

        # convert x_fixed into binary representation
        mask_weights = mask_weights * 2 ** (FB)
        for i in range(len_bits - 1):
            j = len_bits - i - 1
            self.binary_[:, j] = torch.floor(mask_weights % 2).to(device)
            mask_weights = mask_weights // 2

        self.binary_[:, 0] = mask_weights * 0

        # Do masking and convert binary back to fixed point decimal
        self.binary_ = self.binary_.type(torch.float32)
        self.binary_[:, self.index_mask] = self.set_bit
        mask_weights = torch.matmul(self.binary_, coeff)
        mask_weights[index_neg] = - mask_weights[index_neg]

        flattened_weights[self.index] = mask_weights
        self.conv.weight.data = flattened_weights.reshape(np_weight.shape).to(device)
#         print(self.conv.weight.data.cpu().detach().numpy().reshape(-1))
