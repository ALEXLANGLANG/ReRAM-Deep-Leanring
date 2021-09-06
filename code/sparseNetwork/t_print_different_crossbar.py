# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
#from tensorboardX import SummaryWriter
#writer = SummaryWriter()

import torchvision.utils as vutils
import seaborn as sns
import torch.optim as optim
import torch.nn.init as init
import pickle
import pandas as pd
from prune_layer import *


import time
# Custom Libraries
import utils

# Function for Initialization
def weight_init(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


                
def prune_percentage(q = 10):
    global model
    mask_weights()
    array_all_weights = []# flattened array of nonzero values    
    for n,m in model.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy().flatten()
            array_all_weights += list(weight)
        if isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy().flatten()
            array_all_weights += list(weight)
    percentile_value = np.percentile(abs(np.array(array_all_weights)), q) 
    
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
                
def get_weights(fileName, model_tmp):
    list_weights = []
    for n, m in model_tmp.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().detach().numpy()
            list_weights.append(weight)
        if isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().detach().numpy()
            list_weights.append(weight)
    with open(fileName, 'wb') as fp:
        pickle.dump(list_weights, fp)
            
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
        mask_weights(False) #Mask gradients of weights to zero
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



def run_main(args):
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
        from archs.cifar10 import vgg,resnet,AlexNet #,googlenet,mobilenet,dualpath
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
        model = vgg.vgg11_bn().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16_bn().to(device)
    elif args.arch_type == "vgg19":
        model = vgg.vgg19_bn().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(device)
    elif args.arch_type == "googlenet":
        model = googlenet.GoogLeNet().to(device)
    elif args.arch_type == "mobilenet":
        model = mobilenet.MobileNet().to(device)
    elif args.arch_type == "dualpath26":
        model = dualpath.DPN26().to(device)
    else:
        print("\nWrong Model choice\n")
        exit()


    # Copying and Saving Initial State
#
#    model = torch.load(f"/local/data/Xian/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pt")
#    initial_state_dict = copy.deepcopy(model.state_dict())
#    model.apply(weight_init)
#    initial_state_dict = copy.deepcopy(model.state_dict())

    # Optimizer and Loss
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.0, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss().to(device)
#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    
    compress_rate = [10]

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = len(compress_rate)
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    step = 0
    all_loss = np.zeros(args.end_iter,float)
    all_accuracy = np.zeros(args.end_iter,float)
    args.prune_iterations = ITERATION

    print("************************Start ************************")

    for args.arch_type in ["vgg11","vgg16","vgg19","resnet18", "alexnet"]:
        array_all = None
        start = 0
        for args.crossbarSize in [128,64,8]:
            for iter_ in [15,18,22]:
            
                model_tmp = torch.load(f"/local/data/Xian/saves/{args.arch_type}/{args.dataset}/{iter_}_model_{args.prune_type}.pth.tar")
                accuracy = test(model_tmp, test_loader, criterion)
                print(f"\n--- Pruning Level [{iter_}/{ITERATION}]:--- Acc: {accuracy}%")
                res = utils.print_nonzeros(model_tmp,[],args.crossbarSize)
                get_weights("weights_" + args.arch_type + args.dataset + "{iter_}.dat", model_tmp)
        #        list_all.append(res[2])
        #        list_all.append(res[3])
        #        list_all.append(res[4])
                for i in range(2,5):
                    if start == 0:
                        start +=1
                        array_all = np.array(res[i]).reshape(len(res[i]),1)
                    else:
                        array_all = np.hstack((array_all,np.array(res[i]).reshape(len(res[i]),1)))
        print(array_all)
        array_all  = pd.DataFrame(array_all)
        array_all.to_csv("./results/"+ "crossbarSize"+str(args.crossbarSize) + args.arch_type +args.dataset+".csv")
            

    
    
class argument:
    def __init__(self, lr=1.2e-3,batch_size = 128,start_iter = 0,end_iter = 100,print_freq = 1,
                 valid_freq = 1,resume = "store_true",prune_type= "lt",gpu = "0",
                 dataset = "mnist" ,arch_type = "fc1",prune_percent  = 10,prune_iterations = 35, crossbarSize = 128):
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
            self.arch_type = arch_type  # "fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
            self.crossbarSize = crossbarSize

args = argument(arch_type ="resnet18",dataset="cifar10",prune_iterations = 25,prune_percent  = 20,crossbarSize = 128)
run_main(args)










