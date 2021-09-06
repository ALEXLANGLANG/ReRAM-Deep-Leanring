# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os

import torchvision.utils as vutils
import seaborn as sns

import torch.nn.init as init
import pickle
from prune_layer import *
# from tensorboardX import SummaryWriter
# writer = SummaryWriter()
import torch.optim as optim
import time
# Custom Libraries
import utils
def weight_init(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)
        
  
    
def prune_percentage_nonzero(q = 10):
    global model
    mask_weights()
    array_nonzero = []# flattened array of nonzero values    
    for n,m in model.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy().flatten()
            array_nonzero += list(weight[np.nonzero(weight)])
        if isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy().flatten()
            array_nonzero += list(weight[np.nonzero(weight)])  
    percentile_value = np.percentile(abs(np.array(array_nonzero)), q) 
    
    for n,m in model.named_modules():
        if isinstance(m, PrunedConv):
            m.prune_by_percentage(percentile_value = percentile_value)
        if isinstance(m, PruneLinear):
            m.prune_by_percentage(percentile_value = percentile_value)
            
def mask_weights(mask_data = True): 
    global model 
    if mask_data:
        for n, m in model.named_modules():
            if isinstance(m, PrunedConv):
                m.conv.weight.data.mul_(m.mask)
            if isinstance(m, PruneLinear):
                m.linear.weight.data.mul_(m.mask)
    else:
        for n, m in model.named_modules():
            if isinstance(m, PrunedConv):
                m.conv.weight.grad.mul_(m.mask)
            if isinstance(m, PruneLinear):
                m.linear.weight.grad.mul_(m.mask)
            
def initialize_weights(initial_state_dict):
    global model 
    for n,m in model.named_modules():
        if isinstance(m, PrunedConv):
            m.conv.weight.data = m.mask*initial_state_dict[n + '.conv.weight']
            m.conv.weight.bias = initial_state_dict[n + '.conv.bias']
        if isinstance(m, PruneLinear):
            m.linear.weight.data = m.mask*initial_state_dict[n + '.linear.weight']
            m.linear.weight.bias =initial_state_dict[n + '.linear.bias']
            
def reintilize_weights(weight_init):
    global model 
    model.apply(weight_init)
    mask_weights()


# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        
        mask_weights()#Mask data into zero
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()
        mask_weights(False)
        optimizer.step()
    return train_loss.item()


# Function for Testing
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def get_mask_all():
    global model
    mask = []
    for n, m in model.named_modules():
        if isinstance(m, PrunedConv):
            mask += [m.mask.cpu().numpy()]
        if isinstance(m, PruneLinear):
            mask += [m.mask.cpu().numpy()]
    return mask



def train_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ITE=0
    reinit = True if args.prune_type=="reinit" else False

    #Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    
    #Data Loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

                    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('/local/data/Xian', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('/local/data/Xian', train=False, transform=transform)
        from archs.mnist import  LeNet5, fc1, vgg, resnet,AlexNet
    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('/local/data/Xian', train=True, download=True,transform=transform_train)
        testdataset = datasets.CIFAR10('/local/data/Xian', train=False, transform=transform_test)
        from archs.cifar10 import vgg,resnet,AlexNet,googlenet,mobilenet, dualpath
    # If you want to add extra datasets paste here
    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader=torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,num_workers=2)
    test_loader=torch.utils.data.DataLoader(testdataset, batch_size=100, shuffle=False, pin_memory=True,num_workers=2)
    # Importing Network Architecture
    global model
    if args.arch_type == "fc1":
        model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "vgg11":
        model = vgg.vgg11().to(device)  
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16_bn().to(device)
    elif args.arch_type == "vgg19":
        model = vgg.vgg19_bn().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "googlenet":
        model = googlenet.GoogLeNet().to(device)
    elif args.arch_type == "mobilenet":
        model = mobilenet.MobileNet().to(device)
    elif args.arch_type == "dualpath26":
        model = dualpath.DPN26().to(device)
    else:
        print("\nWrong Model choice\n")
        exit()

    model.apply(weight_init)
    norm = "norm"
    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"/local/data/Xian/saves_{norm}/{args.arch_type}/{args.dataset}/")
    torch.save(model, f"/local/data/Xian/saves_{norm}/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pt")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.0, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)


    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    step = 0
    all_loss = np.zeros(args.end_iter,float)
    all_accuracy = np.zeros(args.end_iter,float)
    model_tmp= None
    
    list_compression =[]
    list_all =[]
    list_bestacc =[]
    
    for _ite in range(args.start_iter, ITERATION):
        if not _ite == 0:
            prune_percentage_nonzero(q = args.prune_percent)
            if reinit:
                reintilize_weights(weight_init)
            else:
                initialize_weights(initial_state_dict)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.0, weight_decay=0.0005)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        print(f"\n--- Pruning Level [{_ite}/{ITERATION}]: --- Iterative: unpruned: {0.8**_ite*100.}")

        # Print the table of Nonzeros in each layer
        res = utils.print_nonzeros(model,list_all)
        comp1,list_all = res[0], res[1]
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))
        best_accuracy = 0
        for iter_ in pbar:
            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    model_tmp = copy.deepcopy(model)

            # Training
            loss = train(model, train_loader, optimizer, criterion)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy
            scheduler.step()
            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')
        
        list_bestacc += [best_accuracy]
        list_compression += [comp1]
        utils.checkdir(f"/local/data/Xian/saves/{args.arch_type}/{args.dataset}/")
        torch.save(model_tmp,f"/local/data/Xian/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")

#    utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
#    comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
    
    print(list_compression)
    print(list_bestacc)
#    arr = np.array(list_all).T
#    arr = np.append(arr, np.array([list_bestacc]), axis=0)
    arr = np.array([list_compression])
    arr = np.append(arr, np.array([list_bestacc]), axis=0)
    df = pd.DataFrame(arr)
    df.to_csv(f'{os.getcwd()}/stats/{args.arch_type}{args.dataset}_iterative_{norm}.csv')


class argument:
    def __init__(self, lr=1.2e-3,batch_size = 128,start_iter = 0,end_iter = 100,print_freq = 1,
                 valid_freq = 1,resume = "store_true",prune_type= "lt",gpu = "0",
                 dataset = "mnist" ,arch_type = "fc1",prune_percent  = 10,prune_iterations = 35):
        self.lr = lr
        self.batch_size = batch_size
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.print_freq = print_freq
        self.valid_freq = valid_freq
        self.resume = resume
        self.prune_type = prune_type #reinit
        self.gpu = gpu
        self.dataset = dataset #"mnist | cifar10 | fashionmnist | cifar100"
        self.arch_type = arch_type # "fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121"
        self.prune_percent  = prune_percent 
        self.prune_iterations = prune_iterations 
        
args = argument(end_iter =80,arch_type ="mobilenet",dataset = 'cifar10',prune_percent  = 20,lr= 0.05,prune_iterations = 25)

train_main(args)


















