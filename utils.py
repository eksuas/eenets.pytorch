"""
utilities are defined in this code.
"""
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision import datasets, transforms
from custom_eenet import CustomEENet
from eenet import EENet

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        """reset the meter"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, repetition=1):
        """update the meter"""
        self.val = val
        self.sum += val * repetition
        self.count += repetition
        self.avg = self.sum / self.count


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


def plot_charts(history, args):
    """
    This method plots the history charts.
    """
    fig, axs = plt.subplots(1, 1)
    plt.title('The EENet-8 model trained with the '+args.filename+' loss')
    legend = []
    for key, value in history.items():
        plt.plot(value)
        legend.append(key)
    plt.ylabel('percent')
    plt.xlabel('epochs')
    plt.legend(legend, loc='best')
    #plt.xticks([i for i in range(args.epochs)], [str(i+1) for i in range(args.epochs)])
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig('Results/'+args.filename+'.png')
    plt.clf()


def display_examples(args, model, dataset):
    """
    This method shows the correctly predicted sample of images from dataset.
    Produced table shows the early exit block which classifies that samples.
    """
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
            fig.savefig("Results/exitblock"+str(idx)+".png")


def save_model(args, model):
    """
    This method saves the trained model in pt file.
    """
    directory = '../models/'+args.dataset+'/'+args.model
    if not os.path.exists(directory):
        os.makedirs(directory)

    version = 1
    while os.path.exists(directory+'/v'+str(version)+'.pt'):
        version += 1

    filename = directory+'/v'+str(version)+'.pt'
    torch.save(model, filename)


def adjust_learning_rate(model, optimizer, epoch):
    """
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
