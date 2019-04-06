"""
high level support for doing this and that.
"""
from __future__ import print_function
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from flops_counter import flops_to_string, params_to_string
from utils import AverageMeter
from utils import load_dataset
from utils import adjust_learning_rate
from utils import save_model
#from utils import plot_charts
from init import initializer
from eenet import EENet
from custom_eenet import CustomEENet


def main():
    """Main function of the program.

    The function loads the dataset and calls training and validation functions.
    """
    model, optimizer, args = initializer()
    train_loader, test_loader = load_dataset(args)

    if args.load_model == '':
        # exit distribution of EENet based models
        if isinstance(model, EENet):
            for idx, (flops, params) in enumerate(model.complexity):
                print('exit-block-{}: flops={}, params={}, cost-rate={:.2f}\
                      '.format(idx, flops_to_string(flops),
                               params_to_string(params),
                               flops/model.complexity[-1][0]))
        scheduler = ReduceLROnPlateau(optimizer)

        best = {'acc':None}
        history = {'acc':[], 'loss':[], 'cost':[], 'time':[]}
        for epoch in range(1, args.epochs + 1):
            print('{:2d}:'.format(epoch), end="")

            if args.adjust_lr:
                adjust_learning_rate(model, optimizer, epoch)

            train(args, model, train_loader, optimizer)
            result = validate(args, model, test_loader)
            for key, value in result.items():
                history[key].append(value)
            scheduler.step(result['loss'].avg)
            if best['acc'] is None or result['acc'].avg > best['acc'].avg:
                best = result

        print('The best avg loss: {:.4f}, avg cost:{:.4f}, avg acc:{:.2f}%\
              '.format(best['loss'].avg, best['cost'].avg*100., best['acc'].avg*100.))

        if args.save_model:
            save_model(args, model)

    #plot_charts(history, args)
    #display_examples(args, model, trainset)


def train(args, model, train_loader, optimizer):
    """train the model.

    Arguments are
    * args:         command line arguments entered by user.
    * model:        convolutional neural network model.
    * train_loader: train data loader.
    * optimizer:    optimize the model during training.
    * epoch:        epoch number.

    This trains the model and prints the results of each epochs.
    """
    model.train()
    for data, target in train_loader:
        data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
        optimizer.zero_grad()

        # training settings for EENet based models
        if isinstance(model, (CustomEENet, EENet)):
            pred, conf, cost = model(data)
            cum_pred = [torch.tensor(0.).to(args.device)]*(args.num_ee) + [pred[args.num_ee]]
            cum_cost = [torch.tensor(0.).to(args.device)]*(args.num_ee) \
                     + [torch.tensor(1.).to(args.device)]

            loss = F.nll_loss(cum_pred[args.num_ee].log(), target) \
                 + args.lambda_coef * cum_cost[args.num_ee].mean()
            for i in range(args.num_ee-1, -1, -1):
                cum_pred[i] = conf[i] * pred[i] + (1-conf[i]) * cum_pred[i+1]
                cum_cost[i] = conf[i] * cost[i] + (1-conf[i]) * cum_cost[i+1]
                loss += F.nll_loss(cum_pred[i].log(), target) + args.lambda_coef*cum_cost[i].mean()

        # training settings for other models
        else:
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()


def validate(args, model, val_loader):
    """validate the model.

    Arguments are
    * args:         command line arguments entered by user.
    * model:        convolutional neural network model.
    * val_loader:   validation data loader..

    This validates the model and prints the results of each epochs.
    Finally, it returns average accuracy, loss and comptational cost.
    """
    batch = {'time':AverageMeter(),
             'cost':AverageMeter(),
             'loss':AverageMeter(),
             'acc' :AverageMeter()}
    exit_points = [0]*(args.num_ee+1)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
            # compute output
            start = time.process_time()

            # results of EENet based models
            if isinstance(model, (CustomEENet, EENet)):
                output, idx, cost = model(data)
                batch['time'].update(time.process_time()  - start)
                batch['loss'].update(F.nll_loss(output.log(), target) + args.lambda_coef * cost)
                batch['cost'].update(cost)
                exit_points[idx] += 1

            # results of other models
            else:
                output = model(data)
                batch['time'].update(time.process_time()  - start)
                criterion = nn.CrossEntropyLoss()
                batch['loss'].update(criterion(output, target))

            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            batch['acc'].update(pred.eq(target.view_as(pred)).sum().item())

    # print the results of epoch
    print('Test set avg time: {:.4f}msec; avg loss: {:.4f}; avg acc:{:.2f}%\
          '.format(batch['time'].avg*100., batch['loss'].avg, batch['acc'].avg*100.))

    # detail print for EENet based models
    if isinstance(model, (CustomEENet, EENet)):
        print('\tavg cost: {:.2f}%; exits: <'.format(batch['cost'].avg*100.), end='')
        for i in range(args.num_ee+1):
            print('{:d},'.format(exit_points[i]), end='')
        print('>')

    return batch


if __name__ == '__main__':
    main()
