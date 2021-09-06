import torch
import torch.nn as nn
# from archs.prune_layer import *
from prune_layer import *
# class LeNet5(nn.Module):
#     def __init__(self, num_classes=10):
#         super(LeNet5, self).__init__()
#         self.features = nn.Sequential(
# #             PrunedConv(in_channels=1, out_channels=6,stride=1, padding=0, kernel_size=5)
#             nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(64*14*14, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         print(x.shape)
#         x = self.classifier(x)
#         return x



class LeNet5(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = PrunedConv(in_channels=1, out_channels=64,stride=1, padding=1, kernel_size=3)
        self.BN2d_1 = nn.BatchNorm2d(64)
        self.conv2 = PrunedConv(in_channels=64, out_channels=64,stride=1, padding=1, kernel_size=3)
        self.BN2d_2 = nn.BatchNorm2d(64)
        self.fc1   = PruneLinear(64*14*14, 256)
        self.BN1d_1 = nn.BatchNorm1d(256)
        self.fc2   = PruneLinear(256, 256)
        self.BN1d_2 = nn.BatchNorm1d(256)
        self.fc3   = PruneLinear(256, num_classes)


    def forward(self, x):
        out = F.relu(self.BN2d_1(self.conv1(x)))
        out = F.relu(self.BN2d_2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.BN1d_1(self.fc1(out)))
        out = F.relu(self.BN1d_2(self.fc2(out)))
        x = self.fc3(out)
        
#         out = F.relu((self.conv1(x)))
#         out = F.relu((self.conv2(out)))
#         out = F.max_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         out = F.relu((self.fc1(out)))
#         out = F.relu((self.fc2(out)))
#         x = self.fc3(out)

        return F.log_softmax(x, dim=1)


# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         self.conv1 = PrunedConv(in_channels=1, out_channels=6,stride=1, padding=0, kernel_size=5)
#         self.BN2d_1 = nn.BatchNorm2d(6)
#         self.conv2 = PrunedConv(in_channels=6, out_channels=16,stride=1, padding=0, kernel_size=5)
#         self.BN2d_2 = nn.BatchNorm2d(16)
#         self.fc1   = PruneLinear(16*5*5, 120)
#         self.BN1d_1 = nn.BatchNorm1d(120)
#         self.fc2   = PruneLinear(120, 84)
#         self.BN1d_2 = nn.BatchNorm1d(84)
#         self.fc3   = PruneLinear(84, 10)


#     def forward(self, x):
# #         out = F.relu(self.BN2d_1(self.conv1(x)))
# #         out = F.max_pool2d(out, 2)
# #         out = F.relu(self.BN2d_2(self.conv2(out)))
# #         out = F.max_pool2d(out, 2)
# #         out = out.view(out.size(0), -1)
# #         out = F.relu(self.BN1d_1(self.fc1(out)))
# #         out = F.relu(self.BN1d_2(self.fc2(out)))
# #         x = self.fc3(out)
#         out = F.relu((self.conv1(x)))
#         out = F.max_pool2d(out, 2)
#         out = F.relu((self.conv2(out)))
#         out = F.max_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         out = F.relu((self.fc1(out)))
#         out = F.relu((self.fc2(out)))
#         x = self.fc3(out)

#         return F.log_softmax(x, dim=1)

