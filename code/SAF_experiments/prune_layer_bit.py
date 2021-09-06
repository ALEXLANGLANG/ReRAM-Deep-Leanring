import torch
import torch.nn as nn
import math
import numpy as np
import random

IB = 4
FB = 11
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

    def forward(self, x):
        out = self.linear(x)
        return out
        pass

    def prune_by_percentage(self, mask_bit_position= [1] * 16, q=0.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        mask_bit_position = torch.tensor(mask_bit_position).to(device)
        mask_percentage = q
        np_weight = self.linear.weight.data.clone().detach().to(device)
        unit = 2 ** -FB
        np_weight = torch.clamp(np_weight, max=2 ** IB - 2 ** (-FB), min=-(2 ** IB - 2 ** (-FB)))
        np_weight = torch.round(np_weight / unit) * unit

        len_bits = 1 + FB + IB
        # Do bit mask for each element of weight
        flattened_weights = torch.flatten(np_weight).to(device)
        len_ = len(flattened_weights)
        random.seed(4)
        index = np.array(random.sample(range(0, len_), int(mask_percentage / 100 * len_))).astype(int)

        # Do bit mask for each element of weight
        mask_weights = flattened_weights[index]
        binary_ = torch.tensor(np.array([0] * len_bits * len(index)).reshape((len(index), len_bits))).to(device)
        index_neg = torch.where(mask_weights < 0)[0].to(device).to(device)
        mask_weights[index_neg] = - mask_weights[index_neg]  # change negative into positive

        # convert x_fixed into binary representation
        mask_weights = mask_weights * 2 ** (FB)
        for i in range(len_bits - 1):
            j = len_bits - i - 1
            binary_[:, j] = torch.floor(mask_weights % 2).to(device)
            mask_weights = mask_weights // 2

        binary_[:, 0] = mask_weights * 0

        index_mask = torch.where(mask_bit_position == 0)[0].to(device)
        temp_ = torch.tensor([0.0] * (len(index) * len(index_mask))).to(device)
        temp_ = temp_.type(torch.float32).to(device)
        binary_ = binary_.type(torch.float32).to(device)
        shape_ = binary_[:, index_mask].shape

        binary_[:, index_mask] = temp_.reshape(shape_)

        mask_weights = torch.matmul(binary_, coeff)
        mask_weights[index_neg] = - mask_weights[index_neg]
        flattened_weights[index] = mask_weights
        self.linear.weight.data = flattened_weights.reshape(np_weight.shape).to(device)
        return 0, 0, 0, 0


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

    def forward(self, x):
        out = self.conv(x)
        return out

    def prune_by_percentage(self, mask_bit_position=[1] * 16, q=0.0):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        mask_bit_position = torch.tensor(mask_bit_position).to(device)
        mask_percentage = q
        np_weight = self.conv.weight.data.clone().detach().to(device)
        unit = 2 ** -FB
        np_weight = torch.clamp(np_weight, max=2 ** IB - 2 ** (-FB), min=-(2 ** IB - 2 ** (-FB)))
        np_weight = torch.round(np_weight / unit) * unit

        len_bits = 1 + FB + IB
        # Do bit mask for each element of weight
        flattened_weights = torch.flatten(np_weight).to(device)
        len_ = len(flattened_weights)
        random.seed(4)
        index = np.array(random.sample(range(0, len_), int(mask_percentage / 100 * len_))).astype(int)

        # Do bit mask for each element of weight
        mask_weights = flattened_weights[index]
        binary_ = torch.tensor(np.array([0] * len_bits * len(index)).reshape((len(index), len_bits))).to(device)
        index_neg = torch.where(mask_weights < 0)[0].to(device).to(device)
        mask_weights[index_neg] = - mask_weights[index_neg]  # change negative into positive

        # convert x_fixed into binary representation
        mask_weights = mask_weights * 2 ** (FB)
        for i in range(len_bits - 1):
            j = len_bits - i - 1
            binary_[:, j] = torch.floor(mask_weights % 2).to(device)
            mask_weights = mask_weights // 2

        binary_[:, 0] = mask_weights * 0

        index_mask = torch.where(mask_bit_position == 0)[0].to(device)
        temp_ = torch.tensor([0.0] * (len(index) * len(index_mask))).to(device)
        temp_ = temp_.type(torch.float32).to(device)
        binary_ = binary_.type(torch.float32).to(device)
        shape_ = binary_[:, index_mask].shape

        binary_[:, index_mask] = temp_.reshape(shape_)

        mask_weights = torch.matmul(binary_, coeff)
        mask_weights[index_neg] = - mask_weights[index_neg]
        flattened_weights[index] = mask_weights
        self.conv.weight.data = flattened_weights.reshape(np_weight.shape).to(device)

        return 0, 0, 0, 0
