from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import warnings
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import numpy as np
import pandas as pd
from prune_layer_bit import *
import torch.nn as nn


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = PrunedConv(in_channels=1, out_channels=6,stride=1, padding=0, kernel_size=5)
        self.BN2d_1 = nn.BatchNorm2d(6)
        self.conv2 = PrunedConv(in_channels=6, out_channels=16,stride=1, padding=0, kernel_size=5)
        self.BN2d_2 = nn.BatchNorm2d(16)
        self.fc1   = PruneLinear(16*5*5, 120)
        self.BN1d_1 = nn.BatchNorm1d(120)
        self.fc2   = PruneLinear(120, 84)
        self.BN1d_2 = nn.BatchNorm1d(84)
        self.fc3   = PruneLinear(84, 10)


    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu((self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu((self.fc1(out)))
        out = F.relu((self.fc2(out)))
        x = self.fc3(out)

        return F.log_softmax(x, dim=1)


def get_num_correct(pred,labels):
    return pred.argmax(dim=1).eq(labels).sum().item()


def init_weights(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device =='cuda':
    print("Run on GPU...")
else:
    print("Run on CPU...")
    
    

transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

train_set = datasets.MNIST(root='/local/data/Xian', train=True, download=True,
                   transform=transform)
test_set = datasets.MNIST(root='/local/data/Xian', train=False,download=True,
                   transform=transform)

def train_(train_set,test_set,layer_name, q, batch_size_train = 128, momentum = 0, mask_bit_position = [1]*16, lr = 0.01, epochs = 10):
    torch.manual_seed(1)
    train_loader=torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=False, pin_memory=True,num_workers=2)
    test_loader=torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, pin_memory=True,num_workers=2)


    torch.manual_seed(1)
    network= Net().to(device)
    network.apply(init_weights)
                

    optimizer = optim.SGD(network.parameters(), lr=lr, momentum = momentum, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    acc_train=[]
    acc_test=[]
    acc = 0

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        network.train()
        count_in = 0

        for batch in train_loader: #Get batch

            count_in = count_in + 1
            images,labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            #Do bit maskinn
            for n, m in network.named_modules():
                if  n in layer_name:
                    m.prune_by_percentage(mask_bit_position=mask_bit_position, q = q)

            preds=network(images) #pass batch to network
            correct = get_num_correct(preds, labels)
            loss = criterion(preds,labels) #Calculate loss
            loss.backward() #Calculate gradients
            optimizer.step() #Update weights
            total_correct+=correct
            
        print("epoch: ", epoch,  "total_correct: ", total_correct)
        print("training accuracy: ", total_correct/len(train_set))
        acc_train.append(deepcopy(float(total_correct)/len(train_set)))

        with torch.no_grad():
            correct_test=0
            for batch_test in test_loader: #Get batch
                images_test,labels_test = batch_test
                images_test, labels_test = images_test.to(device), labels_test.to(device)
                preds_test=network(images_test) #pass batch to network
                correct_test += get_num_correct(preds_test, labels_test)
            print("testing accuracy: ", correct_test / len(test_set))
            if epoch == epochs - 1:
                print(correct_test / len(test_set))
                acc = correct_test / len(test_set) 
            acc_test.append(deepcopy(float(correct_test)/len(test_set)))
        scheduler.step()
    
    return acc

# layers
list_layers = [['conv1'],
               ['conv2'],
               ['fc1'],
               ['fc2'],
               ['fc3']
              ]
#Mask the first 8 bits
# mask_bit_position = [1]*16
# mask_bit_position[1] = 0

list_q = [1,5,10]
list_mask = [1]
list_acc = []
for lr in [0.1, 0.05, 0.01,0.001]:
    for layer_name in list_layers:
        for q in list_q:
            for j in list_mask:
                mask_bit_position = [1]*16
                mask_bit_position[j] = 0 
                acc = train_(train_set,test_set,layer_name=layer_name,lr = lr, mask_bit_position = mask_bit_position, q=q,epochs = 10)
                list_acc += [acc]
            print(list_acc)
        
import pandas as pd
list_acc = np.array(list_acc).reshape((len(list_layers),-1))
df = pd.DataFrame (list_acc)
print(df)
print("bit0_lr")
## save to xlsx file
filepath = './results/lent_weights_bit0_lr.csv'



# df.to_csv(filepath, index=False)


