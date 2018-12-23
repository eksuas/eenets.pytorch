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



optional arguments:
  -h, --help                                  show this help message and exit
  --dataset {svhn,cifar10,cifar100,mnist}     choose a dataset from the list. (default: mnist)
  --arch ARCH                                 model architecture. (default: resnet18)
  --epoch EPOCH                               number of epoch to train. (default: 1)
  --batch_size BATCH_SIZE                     batch size. (default: 32)
  --loss {single,combined}                    loss function (default: combined)
  --optimizer {adam,sgd}                      optimizer function. (default: adam)
  --data_aug DATA_AUG                         if data augmentation is performed. (default: False)
  --save_name SAVE_NAME                       model name to save arch, parameters, results etc. (default: )
  --load_model LOAD_MODEL                     file to load model. (default: None)
  --verbose {0,1,2}                           verbosity: 0(silent), 1(prog-bar), or 2(one-line). (default: 1)
  --c C                                       penalty for not exiting of a layer. (default: 0.5)
  --T T                                       threshold of laters' exit gate. (default: 0.5)
  --exit_block {plain,pool,bnpool}            exit block type. (default: plain)
  --figures FIGURES                           if train accuracy and cost figures, charts are created. (default: False)
  --print_summary PRINT_SUMMARY               print the summary of architexture. (default: False)
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
$ python main.py --dataset svhn --load_model snapshots/svhn_eenet18/c0.5_T0.5
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
