Sparse network, SAF and Hardware Saving:
    
    ![alt text](https://github.com/ALEXLANGLANG/ReRAM-Deep-Leanring/blob/master/resources/Mapping_ReRAM.jpg?raw=true)
    
    Interested:
    The relationship between performance of sparse cnn models and SAF
    How many crossbars we can save from the sparse neural network trained based on The Lottery Ticket Hypothesis

    Settings:
    1. Three different datasets: MNIST, CIFAR10, CIFAR100
    2. 6 models: LeNet5, AlexNet,Resnet,VGG, DenseNet, MobileNet
    3. Globally Iterative pruning and one-shot pruning

    One-shot_improved pruning Experiment:
    Add a lower adaptive bound for each fully connected and convolution layer.
    One-shot_proved pruning can achieve much sparser CNN network than original one-shot pruning mentioned in paper "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks".
    It can enable sparse network with sparsity=1% for  VGG11, VGG16, VGG19, AlexNet and sparsity=5% for GoogleNet, MobileNet, DualPath.

    SAF and Sparse network Experiment:
    We were expecting that sparse networks may have better performance on SAF than dense network.
    But the experiments show that sparse networks are much more sensitive to SAF faults than dense network.

    Sparse network and Crossbar Saving in ReRAM
    Implemented a ordering method to reorder the weights of sparse networks
    After reordering, the sparse network global sparsity = 1% can save 96% crossbars (8x8), 89% crossbars (64x64),83% crossbars (128x128)



SAF and CNN:
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

