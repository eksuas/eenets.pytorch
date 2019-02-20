# EENets_PyTorch
This repository contains PyTorch implementation of EENets: Early Exit Convolutional Neural Network. ([the thesis](https://drive.google.com/file/d/1tnLPd2Jiqm3WdVYKYAMv6dF_XpjS9vfu/view) and [the slides](https://drive.google.com/open?id=1IJKm0XygD2yPA1jCSgNdAhApGWM76lFf))

## Getting Started

The codes are developed with python35 environment and tested on Windows and Linux with cuda-9 and cuda-10. The development environment consists of
 * i7-6700HQ CPU processor with 16GB RAM 
 * 2x NVIDIA Tesla P100 16GB.

### Prerequisites

Pytorch environment can be installed via [the website](https://pytorch.org/get-started/locally/).

### Datasets

The models can be trained and tested with MNIST, CIFAR10, SVHN and ImageNet datasets. Data is expected to be in the directory "../data".

```
+/data
    +/mnist
       + training.pt
       + test.pt
    -/cifar10
    -/svhn
    -/imagenet
```

We evaluated MNIST dataset on our custom model EENet-8 (defined in custom_eenet.py) 

### Training

The main.py includes command line arguments, to see them:
```
$ python main.py --help
usage: main.py [-h] [--batch-size N] [--test-batch N] [--epochs N] [--lr LR]
               [--momentum M] [--no-cuda] [--seed S] [--log-interval N]
               [--save-model] [--load-model LOAD_MODEL] [--filters FILTERS]
               [--lamb LAMB] [--num-ee NUM_EE] [--filename FILENAME]
               [--dataset {mnist,cifar10,svhn,imagenet}]
               [--num-classes NUM_CLASSES] [--optimizer {SGD,Adam}]
               [--input-shape INPUT_SHAPE]
               [--distribution {gold_ratio,pareto,fine,linear}]
               [--model {eenet8,eenet18,eenet34,eenet50,eenet101,eenet152,
	                 eenet20,eenet32,eenet44,eenet56,eenet110,resnet18,
			 resnet34,resnet50,resnet101,resnet152,resnet20,
			 resnet32,resnet44,resnet56,resnet110}]

PyTorch MNIST Example

optional arguments:
  -h, --help            			show this help message and exit
  --batch-size N        			input batch size for training (default: 32)
  --test-batch N        			input batch size for testing (default: 1)
  --epochs N            			number of epochs to train (default: 10)
  --lr LR               			learning rate (default: 0.001)
  --momentum M          			SGD momentum (default: 0.5)
  --no-cuda             			disables CUDA training
  --seed S              			random seed (default: 1)
  --log-interval N      			how many batches to wait before logging training status
  --save-model          			save current model
  --load-model LOAD_MODEL			the path for loading and evaluating model
  --filters FILTERS     			initial filter number of custom eenet-8 (default: 2)
  --lamb LAMB           			lambda to arrange the balance between accuracy and cost (default: 1.0)
  --num-ee NUM_EE       			the number of early exit blocks (default: 3)
  --filename FILENAME   			the filename of plots (default: modelChart)
  --dataset {mnist,cifar10,svhn,imagenet}	dataset to be evaluted (default: cifar10)
  --num-classes NUM_CLASSES			the number of classes in the dataset (default: 10)
  --optimizer {SGD,Adam}			optimizer (default: Adam)
  --input-shape INPUT_SHAPE			the shape of dataset (default: (3, 32, 32))
  --distribution {gold_ratio,pareto,		
  	fine,linear}				distribution method of exit blocks (default: fine)		
  --model {eenet8,eenet18,eenet34,eenet50,
  	eenet101,eenet152,eenet20,eenet32,
	eenet44,eenet56,eenet110,resnet18,
	resnet34,resnet50,resnet101,resnet152,
	resnet20,resnet32,resnet44,resnet56,
	resnet110}				model to be evaluated (default: eenet20)						
  
```

Example training command:
```
$ python main.py --model eenet110 --dataset cifar10 --num-ee 8 --epochs 50 --save-model
```

## The Code Contents

ResNet based EENet models are implemented in the "eenet.py" file. Some initialization and parameter parsing works can be found in the "init.py" file. The "main.py" creates and initializes an instance of the spesified model. Details of training and testing procedures are also implemented in this file. The "utils.py" and "flops_counter.py" are helping codes for AverageMeter and counting the number of floating point operations, respectively. "flops_counter.py" are taken from [this repo](https://github.com/sovrasov/flops-counter.pytorch).

## Authors

* **Edanur Demir** - *EENets: Early Exit Convolutional Neural Networks* - [eksuas](https://github.com/eksuas)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

