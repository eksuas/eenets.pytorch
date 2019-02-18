import torch
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
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
        testset  = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)

    elif (args.dataset == 'cifar10'):
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
        testset  = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)

    elif (args.dataset == 'svhn'):
        trainset = datasets.SVHN('./data/svhn', train=True, download=True, transform=transform)
        testset  = datasets.SVHN('./data/svhn', train=False, download=True, transform=transform)

    elif (args.dataset == 'imagenet'):
        trainset = datasets.ImageFolder('./data/imagenet', train=True, download=True, transform=transform)
        testset  = datasets.ImageFolder('./data/imagenet', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, **kwargs)
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
            data = data.view(-1, 1, 28, 28)
            data, target = data.to(device), target.to(device).item()
            output, exit, _ = model(data)
            pred = output.max(1, keepdim=True)[1].item() # get the index of the max log-probability
            if pred == target:
                if len(images[exit][target]) < 10:
                    images[exit][target].append(idx)

        for exit in range(args.num_ee+1):
            fig, axarr = plt.subplots(10, 10)
            for class_id in range(10):
                for example in range(10):
                    axarr[class_id, example].axis('off')
                for example in range(len(images[exit][class_id])):
                    axarr[class_id, example].imshow(dataset[images[exit][class_id][example]][0].view(28, 28))
            fig.savefig("Results/exitblock"+str(exit)+".png")
