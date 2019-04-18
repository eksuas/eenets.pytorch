"""
utilities are defined in this code.
"""
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from torchvision import datasets, transforms
from custom_eenet import CustomEENet
from eenet import EENet


def load_dataset(args):
    """dataset loader.

    This loads the dataset.
    """
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.device == 'cuda' else {}

    if args.dataset == 'mnist':
        root = '../data/mnist'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    elif args.dataset == 'cifar10':
        root = '../data/cifar10'
        trainset = datasets.CIFAR10(root=root, train=True, download=True,\
            transform=transforms.Compose([\
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

        testset = datasets.CIFAR10(root=root, train=False, download=True,\
            transform=transforms.Compose([\
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

    elif args.dataset == 'svhn':
        #def target_transform(target):
        #    return target[0]-1

        root = '../data/svhn'
        transform = transforms.Compose([\
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.SVHN(root=root, split='train', download=True, transform=transform)
                    #,target_transform=target_transform)
        testset = datasets.SVHN(root=root, split='test', download=True, transform=transform)
                    #,target_transform=target_transform)

    elif args.dataset == 'imagenet':
        root = '../data/imagenet'
        trainset = datasets.ImageFolder(root=root+'/train', transform=transforms.Compose([\
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

        testset = datasets.ImageFolder(root=root+'/val', transform=transforms.Compose([\
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    elif args.dataset == 'tiny-imagenet':
        create_val_img_folder()
        root = '../data/tiny-imagenet'
        trainset = datasets.ImageFolder(root=root+'/train', transform=transforms.Compose([\
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))

        testset = datasets.ImageFolder(root=root+'/val/images', transform=transforms.Compose([\
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,\
                    shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch,\
                    shuffle=False, **kwargs)
    return train_loader, test_loader


def create_val_img_folder():
    """
    This method is responsible for separating validation images into separate sub folders.
    """
    val_dir = '../data/tiny-imagenet/val'
    img_dir = os.path.join(val_dir, 'images')

    file = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = file.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    file.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


def plot_history(args, history):
    """plot figures

    Argument is
    * history:  history to be plotted.

    This plots the history in a chart.
    """
    directory = '../results/'+args.dataset+'/'+args.model
    if args.num_ee > 0:
        directory += '/ee'+str(args.num_ee)+'_'+args.distribution
    if not os.path.exists(directory):
        os.makedirs(directory)

    data = pd.DataFrame(history)

    title = 'loss of '+args.model+' on '+args.dataset
    xtick = data['epoch']
    yticks = data[['train_loss', 'val_loss']]
    labels = ('epochs', 'loss')
    filename = directory+'/loss_figure.png'
    plot_chart(title, xtick, yticks, labels, filename)

    title = 'val. accuracy and cost rate of '+args.model+' on '+args.dataset
    xtick = data['epoch']
    yticks = data[['acc', 'cost']]
    labels = ('epochs', 'percent')
    filename = directory+'/acc_cost_figure.png'
    plot_chart(title, xtick, yticks, labels, filename)

    title = 'val. accuracy vs flops of '+args.model+' on '+args.dataset
    xtick = data['flop']
    yticks = data[['acc']]
    labels = ('flops', 'accuracy')
    filename = directory+'/acc_vs_flop_figure.png'
    plot_chart(title, xtick, yticks, labels, filename)


def plot_chart(title, xtick, yticks, labels, filename):
    """draw chart

    Arguments are
    * title:     title of the chart.
    * xtick:     array that includes the xtick values.
    * yticks:    array that includes the ytick values.
    * labels:    labels of x and y axises.
    * filename:  filename of the chart.

    This plots the history in a chart.
    """
    plt.title(title)
    xerr = [stats.sem(x) if isinstance(x, list) else 0 for x in xtick]
    xtick = [np.mean(x) if isinstance(x, list) else x for x in xtick]
    plt.xticks(xtick)

    legend = []
    for key, ytick in yticks.items():
        legend.append(key)
        yerr = [stats.sem(y) for y in ytick]
        ytick = [np.mean(y) for y in ytick]
        plt.errorbar(xtick, ytick, xerr=xerr, yerr=yerr, capsize=3)

    xlabel, ylabel = labels
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc='best')
    plt.savefig(filename)
    plt.clf()
    print('The figure is plotted under \'{}\''.format(filename))


def display_examples(args, model, dataset):
    """display examples

    Arguments are
    * model:    model object.
    * dataset:  dataset loader object.

    This method shows the correctly predicted sample of images from dataset.
    Produced table shows the early exit block which classifies that samples.

    """
    directory = '../results/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    images = [[[] for j in range(10)] for i in range(args.num_ee+1)]
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataset):
            data = data.view(-1, *args.input_shape)
            data, target = data.to(args.device), target.to(args.device).item()
            output, idx, _ = model(data)
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1].item()
            if pred == target:
                if len(images[idx][target]) < 10:
                    images[idx][target].append(idx)

        for idx in range(args.num_ee+1):
            fig, axarr = plt.subplots(10, 10)
            for class_id in range(10):
                for example in range(10):
                    axarr[class_id, example].axis('off')
                for example in range(len(images[idx][class_id])):
                    axarr[class_id, example].imshow(
                        dataset[images[idx][class_id][example]][0].view(args.input_shape[1:]))
            fig.savefig(directory+'exitblock'+str(idx)+'.png')


def save_model(args, model, epoch, best=False):
    """save model

    Arguments are
    * model:    model object.
    * best:     version number of the best model object.

    This method saves the trained model in pt file.
    """
    directory = '../models/'+args.dataset+'/'+args.model
    if isinstance(model, (EENet, CustomEENet)):
        directory += '/ee'+str(args.num_ee)+'_'+args.distribution

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory+'/model'
    if best is False:
        torch.save(model, filename+'.v'+str(epoch)+'.pt')
    else:
        train_files = os.listdir(directory)
        for train_file in train_files:
            if not train_file.endswith('.v'+str(epoch)+'.pt'):
                os.remove(os.path.join(directory, train_file))
        os.rename(filename+'.v'+str(epoch)+'.pt', filename+'.pt')


def adjust_learning_rate(model, optimizer, epoch):
    """adjust learning rate

    Arguments are
    * optimizer:   optimizer object.
    * epoch:       the current epoch number when the function is called.

    This method adjusts the learning rate of training.
    """
    learning_rate = 0.1
    if epoch > 150:
        learning_rate = 0.01
    if epoch > 250:
        learning_rate = 0.001
    if isinstance(model, (EENet, CustomEENet)):
        learning_rate *= 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def print_validation(args, batch, exit_points=None):
    """print validation results

    Arguments are
    * batch:         validation batch results.
    * exit_points:   the number of samples exiting from the specified exit blocks.

    This method prints the results of validation.
    """
    # print the validation results of epoch
    print('     Test avg time: {:.4f}msec; avg val_loss: {:.4f}; avg val_acc:{:.2f}%'
          .format(np.mean(batch['time'])*100.,
                  np.mean(batch['val_loss']),
                  np.mean(batch['acc'])*100.))

    # detail print for EENet based models
    if exit_points is not None:
        print('\tavg val_cost: {:.2f}%; exits: <'.format(np.mean(batch['cost'])*100.), end='')
        for i in range(args.num_ee+1):
            print('{:d},'.format(exit_points[i]), end='')
        print('>')
