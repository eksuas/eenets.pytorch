from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from utils import AverageMeter
from EENets import EENet
from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--filters', type=int, default=2,
                        help='initial filter number of the model')
    parser.add_argument('--lamb', type=float, default=1.0,
                        help='lambda to arrange the balance between accuracy and cost')
    parser.add_argument('--num_ee', type=int, default=2,
                        help='the number of early exit blocks')
    parser.add_argument('--filename', type=str, default='modelChart',
                        help='the filename of plots')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
    testset  = datasets.MNIST('./data/mnist', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = EENet(args.filters).to(device)
    #summary(model, (1, 28, 28))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    best = {'acc':0}
    history = {'acc':[], 'loss':[], 'cost':[]}
    for epoch in range(1, args.epochs + 1):
        print('{:2d}:'.format(epoch), end ="")
        train(args, model, device, train_loader, optimizer, epoch)
        result = validate(args, model, device, test_loader)
        for key, value in result.items():
            history[key].append(value)
        scheduler.step(result['loss'])
        # save model
        if result['acc'] > best['acc']:
            best = result
    print('The best avg loss: {:.4f}, avg cost:{:.4f}, avg acc:{:.2f}%'.format(best['loss'], best['cost']*100., best['acc']*100.))
    plotCharts(history, args)
    display(args, model, device, trainset)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y, h = model(data)
        c = (torch.tensor(0.08).to(device), torch.tensor(0.26).to(device), torch.tensor(1.00).to(device))
        Y = [0]*(args.num_ee) + [y[args.num_ee]]
        C = [0]*(args.num_ee) + [c[args.num_ee]]

        loss = F.nll_loss(torch.log(Y[args.num_ee]), target) + args.lamb * torch.mean(C[args.num_ee])
        for i in range(args.num_ee-1,-1,-1):
            Y[i] = h[i] * y[i] + (1-h[i]) * Y[i+1]
            C[i] = h[i] * c[i] + (1-h[i]) * C[i+1]
            loss += F.nll_loss(torch.log(Y[i]), target) + args.lamb * torch.mean(C[i])

        #loss = F.nll_loss(torch.log(Y[0]), target) + args.lamb * torch.mean(C[0])
        loss.backward()
        optimizer.step()

def validate(args, model, device, val_loader):
    btime = AverageMeter()
    bcost = AverageMeter()
    bloss = AverageMeter()
    bacc = AverageMeter()
    correct = 0
    exit_points = [0]*(args.num_ee+1)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            # compute output
            start = time.process_time()
            output, id, cost = model(data)
            btime.update(time.process_time()  - start)
            exit_points[id] += 1
            bloss.update(F.nll_loss(torch.log(output), target) + args.lamb * cost)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            bacc.update(pred.eq(target.view_as(pred)).sum().item())
            bcost.update(cost)

    print('Test set avg time: {:.4f}msec; avg loss: {:.4f}; avg cost: {:.4f}; exits: <{:d},{:d},{:d}>; avg acc:{:.2f}%'.format(
        btime.avg*100., bloss.avg, bcost.avg*100., exit_points[0], exit_points[1], exit_points[2], bacc.avg*100.))
    return {'acc':bacc.avg, 'loss':bloss.avg, 'cost':bcost.avg}

def display(args, model, device, dataset):
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

if __name__ == '__main__':
    main()
