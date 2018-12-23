# EENets_pytorch
This repository contains Pytroch implementation of EENets: Early Exit Convolutional Neural Network. ([the details and keras implementation](https://github.com/eksuas/EENet) and [the slides](https://docs.google.com/presentation/d/1c-C3MewSl3aXxxits3Vm7k2z5h2RVUp5U2ypb9Xq9Q0/edit?usp=sharing))

## Getting Started

The codes are developed with python35 environment on Windows 10. The development environment consists of 
 * i7-6700HQ CPU processor with 16GB RAM 
 * NVIDIA GeForce GTX 950M 4GB

### Prerequisites

Pytorch environment can be installed via [the website](https://pytorch.org/get-started/locally/).

### Datasets

The models can be trained and tested with MNIST dataset.
training.pt and test.pt should be under "./data/mnist" the folder like:
```
./data/mnist
   + training.pt
   + test.pt
```

Note that MNIST dataset cannot be used in the original ResNet models because their architecture is designed for 32x32 dimensial inputs.
Similarly, ResNet based EENet models do not support MNIST dataset. 

### Training

The main.py includes command line arguments, to see all of them:
```
$ python main.py --help
usage: main.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N]
               [--lr LR] [--momentum M] [--no-cuda] [--seed S]
               [--log-interval N] [--save-model] [--filters N]

PyTorch MNIST Example

optional arguments:
  -h, --help           show this help message and exit
  --batch-size N       input batch size for training (default: 32)
  --test-batch-size N  input batch size for testing (default: 1)
  --epochs N           number of epochs to train (default: 10)
  --lr LR              learning rate (default: 0.01)
  --momentum M         SGD momentum (default: 0.5)
  --no-cuda            disables CUDA training
  --seed S             random seed (default: 1)
  --log-interval N     how many batches to wait before logging training status
  --save-model         For Saving the current Model
  --filters N          initial filter number of the model
```

Example training command:
```
$ python main.py --dataset svhn --arch eenet18 --epoch 200 --save_name svhn_eenet18
```
The weights, architecture and logs of the trained model are saved under the folder "snapshots/<save_name>" and can be loaded later.
Trained models are automaticaly tested after training process. The test results are also saved under the same folder.

### Testing

A trained model can be loaded and tested later. Example command:
```
$ python main.py --filters 2 --epochs 30
```
The test or prediction results are saved in the file "<load_model>/prediction.txt".
In this file, it can be seen that which example are classified at which layer and whether it is classified correctly or not.
The accuracy and cost metrics of testing are prompt to the command line.

## The Code Contents

ResNet based EENet model classes are implemented in the "nn.py" file. 
The "main.py" creates and initializes a model by calling nn methods.
The "callback.py" and "figures.py" are helping codes for training custom callback functions and drawing figures and charts, respectively.

## Authors

* **Edanur Demir** - *EENets: Early Exit Convolutional Neural Networks* - [eksuas](https://github.com/eksuas)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
