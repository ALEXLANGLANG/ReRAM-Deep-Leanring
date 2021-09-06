

# Sparse network and ReRAM Hardware Saving:
    
ReRAM is a promising hardware architecture to accelerate the AI training and inference. But ReRAM needs so many crossbars to achieve hight speed. We want to find a method to train neural networks with the smallest number of crossbars. Sparse neural networks trained based on The Lottery Ticket Hypothesis enable us to save even 90% of crossbars for training. We trained several CNN models such as LeNet5, AlexNet,Resnet,VGG on datasets like MNIST, CIFAR10, CIFAR100, using Globally Iterative magnitude pruning (IMP) and one-shot pruning.  We also designed an improved verisno of one-shot prunning, which have similar performance as IMP. 


## One-shot_improved pruning Experiment:
Some layers of network will be completely pruned using one-shot pruning. In order to keep the siginificant weights of each layer, we added a lower adaptive bound for each fully connected and convolution layer. 
From the figure below, we can see one-shot_improved pruning can achieve much sparser CNN network than original one-shot pruning mentioned in paper "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks".
It can enable sparse network with sparsity=1% for  VGG11, VGG16, VGG19, AlexNet and sparsity=5% for GoogleNet, MobileNet, DualPath.
![one-shot_pruning](https://user-images.githubusercontent.com/49976598/132153853-4a084058-245b-4c4c-9373-25f6ba618fc7.jpg)


## Sparse network and Reordering method saved 90% of crossbars in ReRAM
We Implemented a ordering method to reorder the weights of sparse networks. The figure below is an example where we reordered the layer7 of VGG19 and we saved 94 crossbars in this layer.
![reordering_weights](https://user-images.githubusercontent.com/49976598/132153863-d1ad9cff-cd63-484c-856d-6a863de667c1.jpg)

From the table, the sparse VGG19 global sparsity = 1% can save 96% crossbars (8x8), 89% crossbars (64x64),83% crossbars (128x128), after reordering. This method also works for Resnet, AlexNet, etc.
![hardware_saving](https://user-images.githubusercontent.com/49976598/132155734-15b62876-0fe7-4f85-bcc1-72c8125e8efe.jpg)



SAF and CNN:


## SAF and Sparse network Experiment:
We were expecting that sparse networks may have better performance on SAF than dense network.
But the experiments show that sparse networks are much more sensitive to SAF faults than dense network.

Interested:
This experiments is exploring the performance of models when there are SAF

Settings:
1. SAF happens at different bit position: sign bit, the most significant and the least bit
2. SAF happens at different layers: convolution layers, fully connected layers
4. SAF happens at different models: Lenet5, VGG11, VGG16, etc.

Main Steps:
1. Weights Quantization
2. Mask different position of weights to simulate "stuck at fault" (SAF)
3. Train convolutional neural networks with SAF weights

Observations:
1. For both VGG11 and Lenet5, the first layer is less sensitive to bit error
2. Normalization Layer helps but cannot solve the bit error problem
3. The test accuracy will become random guess even for only 5 percentage of weights are stuck at fault for a particular layer

