from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import time

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torchsummary import summary
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
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
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
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/mnist', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/mnist', train=False, transform=transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = EENet().to(device)
    summary(model, (1, 28, 28))
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    best = (0,0,0)
    num_ee = 2
    for epoch in range(1, args.epochs + 1):
        print(epoch, end =": ")
        train(args, model, num_ee, device, train_loader, optimizer, epoch)
        acc, loss, cost = validate(args, model, num_ee, device, test_loader)
        scheduler.step(loss)
        # save model
        if acc > best[0]:
            best = (acc, loss, cost)
        #test(args, model, device, test_loader)
    print('Best avg loss: {:.4f}, avg cost: {:.4f}, Accuracy:{:.2f}%'.format(best[1], best[2], best[0]))

def train(args, model, num_ee, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y, h = model(data)
        c = (torch.tensor(0.08), torch.tensor(0.26), torch.tensor(1.00))
        Y = [0]*(num_ee) + [y[num_ee]]
        C = [0]*(num_ee) + [c[num_ee]]

        loss = F.nll_loss(torch.log(Y[num_ee]), target) + torch.mean(C[num_ee])
        for i in range(num_ee-1,-1,-1):
            Y[i] = h[i] * y[i] + (1-h[i]) * Y[i+1]
            C[i] = h[i] * c[i] + (1-h[i]) * C[i+1]
            loss += F.nll_loss(torch.log(Y[i]), target) + torch.mean(C[i])
        #loss = criterion(Y[0], target) + torch.mean(C[0])
        loss.backward()
        optimizer.step()

def validate(args, model, num_ee, device, val_loader):
    btime = AverageMeter()
    bcost = AverageMeter()
    bloss = AverageMeter()
    bacc = AverageMeter()
    correct = 0
    exit_points = [0]*(num_ee+1)
    # switch to evaluate mode
    model.eval()
    end = time.time()

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            # compute output
            output, id, cost = model(data)
            exit_points[id] += 1
            bloss.update(F.nll_loss(torch.log(output), target) + cost)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            bacc.update(pred.eq(target.view_as(pred)).sum().item())
            # measure elapsed time
            btime.update(time.time() - end)
            end = time.time()
            bcost.update(cost)

    print('Test set avg time: {:.4f}msec Avg loss: {:.4f}, Avg cost: {:.4f}, Exits: <{:d},{:d},{:d}>, Accuracy:{:.2f}%'.format(
        btime.avg*100., bloss.avg, bcost.avg, exit_points[0], exit_points[1], exit_points[2], bacc.avg*100.))
    return bacc.avg, bloss.avg, bcost.avg

if __name__ == '__main__':
    main()
