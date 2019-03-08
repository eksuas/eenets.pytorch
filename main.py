from __future__ import print_function
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.ticker import MaxNLocator
from torchsummary import summary
from flops_counter import *
#from thop import profile
from init import *
from utils import *

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    Model, Optimizer, args, optimizer_kwargs = initializer()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = load_dataset(args, use_cuda)


    if args.load_model is not '':
        model = torch.load(args.load_model).to(device)

    else:
        kwargs = vars(args)
        model = Model(**kwargs).to(device)

        # exit distribution of EENet based models
        if isinstance(model, EENet):
            for flops, params in  model.complexity:
                print('flops={}, params={}, rate={:.2f}'.format(flops_to_string(flops),
                                                         params_to_string(params),
                                                         flops/model.complexity[-1][0]))
        #summary(model, (1, 28, 28))
        optimizer = Optimizer(model.parameters(), **optimizer_kwargs)
        scheduler = ReduceLROnPlateau(optimizer)

        best = {'acc':0}
        history = {'acc':[], 'loss':[], 'cost':[]}
        for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(model, optimizer, epoch)
            print('{:2d}:'.format(epoch), end ="")
            train(args, model, device, train_loader, optimizer, epoch)
            result = validate(args, model, device, test_loader)
            for key, value in result.items():
                history[key].append(value)
            scheduler.step(result['loss'])
            # save model
            if result['acc'] > best['acc']:
                best = result

        print('The best avg loss: {:.4f}, avg cost:{:.4f}, avg acc:{:.2f}%'.format(best['loss'],
            best['cost']*100., best['acc']*100.))

        if args.save_model:
            save_model(args, model)

    #plotCharts(history, args)
    #display_examples(args, model, device, trainset)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.int64)
        optimizer.zero_grad()

        # training settings for EENet based models
        if isinstance(model, EENet) or isinstance(model, CustomEENet):
            y, h, c = model(data)
            Y = [torch.tensor(0.).to(device)]*(args.num_ee) + [y[args.num_ee]]
            C = [torch.tensor(0.).to(device)]*(args.num_ee) + [torch.tensor(1.).to(device)]

            loss = F.nll_loss(torch.log(Y[args.num_ee]), target) + \
                   args.lamb * torch.mean(C[args.num_ee])
            for i in range(args.num_ee-1,-1,-1):
                Y[i] = h[i] * y[i] + (1-h[i]) * Y[i+1]
                C[i] = h[i] * c[i] + (1-h[i]) * C[i+1]
                loss += F.nll_loss(torch.log(Y[i]), target) + args.lamb * torch.mean(C[i])

        # training settings for other models
        else:
            output = model(data)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, target)

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
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            # compute output
            start = time.process_time()

            # results of EENet based models
            if isinstance(model, EENet) or isinstance(model, CustomEENet):
                output, id, cost = model(data)
                btime.update(time.process_time()  - start)
                exit_points[id] += 1
                bloss.update(F.nll_loss(torch.log(output), target) + args.lamb * cost)
                bcost.update(cost)

            # results of other models
            else:
                output = model(data)
                btime.update(time.process_time()  - start)
                criterion = torch.nn.CrossEntropyLoss()
                bloss.update(criterion(output, target))

            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            bacc.update(pred.eq(target.view_as(pred)).sum().item())

    # print the results of epoch
    print('Test set avg time: {:.4f}msec; avg loss: {:.4f}; avg acc:{:.2f}%'.format(btime.avg*100.,
        bloss.avg, bacc.avg*100.))

    # detail print for EENet based models
    if isinstance(model, EENet) or isinstance(model, CustomEENet):
        print('\tavg cost: {:.2f}%; exits: <'.format(bcost.avg*100.), end='')
        for i in range(args.num_ee+1):
            print('{:d},'.format(exit_points[i]), end='')
        print('>')

    return {'acc':bacc.avg, 'loss':bloss.avg, 'cost':bcost.avg}


if __name__ == '__main__':
    main()
