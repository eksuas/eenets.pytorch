# EENets: Early Exit Convolutional Neural Networks
This repository contains PyTorch implementation of EENets: Early Exit Convolutional Neural Network. ([the thesis](https://drive.google.com/file/d/1tnLPd2Jiqm3WdVYKYAMv6dF_XpjS9vfu/view) and [the slides](https://drive.google.com/open?id=1IJKm0XygD2yPA1jCSgNdAhApGWM76lFf))

## Getting Started

The codes are developed with python35 environment and tested on Windows and Linux with cuda-9 and cuda-10. The development environment consists of
 * i7-6700HQ CPU processor with 16GB RAM
 * NVIDIA Tesla P100 16GB.

### Prerequisites

Pytorch environment can be installed via [the website](https://pytorch.org/get-started/locally/).

### Datasets

The models can be trained and tested with MNIST, CIFAR10, SVHN, Tiny-ImageNet and ImageNet datasets. Datasets are expected to be in the directory "../data".

```
+/data
    -/cifar10
    -/imagenet
    -/mnist
    -/svhn
    +/tiny-imagenet
       - /train
       - /val    
```

### Training

The main.py includes command line arguments, to see them:
```
$ python main.py --help
usage: main.py [-h] [--batch-size N] [--test-batch N] [--epochs N] [--lr N]
               [--adaptive-lr] [--momentum N] [--weight-decay N] [--no-cuda]
               [--seed N] [--log-interval N] [--no-save-model] [--load-model S]
               [--filters N] [--lambda-coef N] [--num-ee N]
               [--dataset {mnist,cifar10,svhn,imagenet,tiny-imagenet}]
               [--optimizer {SGD,Adam}]
               [--distribution {gold_ratio,pareto,fine,linear}]
               [--exit-type {plain,pool,bnpool}]
               [--model {eenet8,eenet18,eenet34,eenet50,eenet101,eenet152,eenet20,
                 eenet32,eenet44,eenet56,eenet110,resnet18,resnet34,resnet50,
                 resnet101,resnet152,resnet20,resnet32,resnet44,resnet56,resnet110}]

PyTorch MNIST Example

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 32)
  --test-batch N        input batch size for testing (default: 1)
  --epochs N            number of epochs to train (default: 150)
  --lr N                learning rate (default: 0.001)
  --adaptive-lr         adjust the learning rate
  --momentum N          SGD momentum (default: 0.9)
  --weight-decay N      weight decay for optimizers (default: 0.0001)
  --no-cuda             disables CUDA training
  --seed N              random seed (default: 1)
  --log-interval N      how many epochs to wait before logging training status
                        (default: 1)
  --no-save-model       do not save the current model
  --load-model S        the path for loading and evaluating model
  --filters N           initial filters of custom eenet-8 (default: 2)
  --lambda-coef N       lambda to arrange the balance between accuracy and
                        cost (default: 1.0)
  --num-ee N            the number of early exit blocks (default: 2)
  --dataset {mnist,cifar10,svhn,imagenet,tiny-imagenet}
                        dataset to be evaluated (default: cifar10)
  --optimizer {SGD,Adam}
                        optimizer (default: Adam)
  --distribution {gold_ratio,pareto,fine,linear}
                        distribution method of exit blocks (default: fine)
  --exit-type {plain,pool,bnpool}
                        Exit block type.
  --model {eenet8,eenet18,eenet34,eenet50,eenet101,eenet152,
           eenet20,eenet32,eenet44,eenet56,eenet110,resnet18,
           resnet34,resnet50,resnet101,resnet152,resnet20,
           resnet32,resnet44,resnet56,resnet110}
                        model to be evaluated (default: eenet20)
```

Example training command:
```
$ python main.py --model eenet110 --dataset cifar10 --num-ee 8 --epochs 50
```

## The Code Contents

ResNet based EENet models are implemented in the "eenet.py" file. Some initialization and parameter parsing works can be found in the "init.py" file. The "main.py" creates and initializes an instance of the specified model. Details of training and testing procedures are also implemented in this file. The "utils.py" includes the helping methods. Finally, "flops_counter.py" counts the number of floating point operations. "flops_counter.py" are taken from [this repo](https://github.com/sovrasov/flops-counter.pytorch).

## Authors

* **Edanur Demir** - *EENets: Early Exit Convolutional Neural Networks* - [eksuas](https://github.com/eksuas)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
