import torch
import os
from torchvision import datasets, transforms

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_dataset(args, use_cuda):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if (args.dataset == 'mnist'):
        root = '../data/mnist'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        testset  = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    elif (args.dataset == 'cifar10'):
        root = '../data/cifar10'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        testset  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    elif (args.dataset == 'svhn'):
        root = '../data/svhn'
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = datasets.SVHN(root=root, split='train', download=True, transform=transform)
        testset  = datasets.SVHN(root=root, split='test', download=True, transform=transform)

    elif (args.dataset == 'imagenet'):
        root = '../data/imagenet'
        trainset = datasets.ImageFolder(root=root+'/train', transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

        testset  = datasets.ImageFolder(root=root+'/val', transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                    shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(testset, batch_size=args.test_batch,
                    shuffle=False, **kwargs)
    return train_loader, test_loader


def plotCharts (history, args):
    fig, axs = plt.subplots(1,1)
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


def display_examples(args, model, device, dataset):
    images = [[[] for j in range(10)] for i in range(args.num_ee+1)]
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataset):
            data = data.view(-1, *args.input_shape)
            data, target = data.to(device), target.to(device).item()
            output, exit, _ = model(data)
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1].item()
            if pred == target:
                if len(images[exit][target]) < 10:
                    images[exit][target].append(idx)

        for exit in range(args.num_ee+1):
            fig, axarr = plt.subplots(10, 10)
            for class_id in range(10):
                for example in range(10):
                    axarr[class_id, example].axis('off')
                for example in range(len(images[exit][class_id])):
                    axarr[class_id, example].imshow(
                        dataset[images[exit][class_id][example]][0].view(args.input_shape[1:]))
            fig.savefig("Results/exitblock"+str(exit)+".png")


def save_model(args, model):
    directory = '../models/'+args.model
    if not os.path.exists(directory):
        os.makedirs(directory)

    version = 1
    while os.path.exists(directory+'/v'+str(version)+'.pt'):
        version += 1

    filename = directory+'/v'+str(version)+'.pt'
    torch.save(model, filename)
