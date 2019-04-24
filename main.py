"""
high level support for doing this and that.
"""
from __future__ import print_function
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from utils import load_dataset
from utils import adaptive_learning_rate
from utils import save_model
from utils import save_history
from utils import plot_history
from utils import print_validation
from init import initializer
from eenet import EENet
from custom_eenet import CustomEENet


def main():
    """Main function of the program.

    The function loads the dataset and calls training and validation functions.
    """
    model, optimizer, args = initializer()
    train_loader, test_loader = load_dataset(args)
    best = {}
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('{:3d}: '.format(epoch), end='')
        result = {'epoch':epoch}

        # use adaptive learning rate
        if args.adaptive_lr:
            adaptive_learning_rate(model, optimizer, epoch)
        result.update(train(args, model, train_loader, optimizer))

        # validate and keep history at each log interval
        if epoch % args.log_interval == 0:
            result.update(validate(args, model, test_loader))
            save_history(args, result)
            if not best or result['val_loss'] < best['val_loss']:
                best = result
                best_epoch = epoch

        # save model parameters
        if not args.no_save_model:
            save_model(args, model, epoch)

    # print the best validation result
    print('\nThe best avg val_loss: {:.4f}, avg val_cost: {:.2f}%, avg val_acc: {:.2f}%\n'
          .format(best['val_loss'], best['cost'], best['acc']))

    # save the model giving the best validation results as a final model
    if not args.no_save_model:
        save_model(args, model, best_epoch, True)
    plot_history(args)


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
    losses = []
    model.train()
    for data, target in train_loader:
        data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
        optimizer.zero_grad()

        # training settings for EENet based models
        if isinstance(model, (CustomEENet, EENet)):
            pred, conf, cost = model(data)

            # Calculate cumulative prediction and cost during training
            cum_pred = [None] * args.num_ee + [pred[args.num_ee]]
            cum_cost = [None] * args.num_ee + [torch.tensor(1.0).to(args.device)]
            loss = F.nll_loss(cum_pred[-1].log(), target) + args.lambda_coef*cum_cost[-1].mean()
            for i in range(args.num_ee-1, -1, -1):
                cum_pred[i] = conf[i] * pred[i] + (1-conf[i]) * cum_pred[i+1]
                cum_cost[i] = conf[i] * cost[i] + (1-conf[i]) * cum_cost[i+1]
                loss += F.nll_loss(cum_pred[i].log(), target) + args.lambda_coef*cum_cost[i].mean()

        # training settings for other models
        else:
            pred = model(data)
            loss = F.cross_entropy(pred, target)

        losses.append(float(loss))
        loss.backward()
        optimizer.step()

    # print the training results of epoch
    result = {'train_loss': round(np.mean(losses), 4),
              'train_loss_sem': round(stats.sem(losses), 2)}

    print('Train avg loss: {:.4f}'.format(result['train_loss']))
    return result


def validate(args, model, val_loader):
    """validate the model.

    Arguments are
    * args:         command line arguments entered by user.
    * model:        convolutional neural network model.
    * val_loader:   validation data loader..

    This validates the model and prints the results of each epochs.
    Finally, it returns average accuracy, loss and comptational cost.
    """
    batch = {'time':[], 'cost':[], 'flop':[], 'acc':[], 'val_loss':[]}
    exit_points = [0]*(args.num_ee+1)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
            # compute output
            start = time.process_time()

            # results of EENet based models
            if isinstance(model, (EENet, CustomEENet)):
                pred, idx, cost = model(data)
                elapsed_time = time.process_time()  - start
                loss = F.nll_loss(pred.log(), target) + args.lambda_coef * cost
                flop = cost * model.complexity[-1][0]
                exit_points[idx] += 1

            # results of other models
            else:
                pred = model(data)
                elapsed_time = time.process_time()  - start
                loss = F.cross_entropy(pred, target)
                flop, cost = model.complexity[-1][0], 1.0
                exit_points = None

            # get the index of the max log-probability
            pred = pred.max(1, keepdim=True)[1]
            acc = pred.eq(target.view_as(pred)).sum().item()
            batch['acc'].append(acc*100.)
            batch['time'].append(elapsed_time)
            batch['cost'].append(cost*100.)
            batch['flop'].append(flop)
            batch['val_loss'].append(float(loss))

    print_validation(args, batch, exit_points)

    result = {}
    for key, value in batch.items():
        result[key] = round(np.mean(value), 4)
        result[key+'_sem'] = round(stats.sem(value), 2)
    return result


if __name__ == '__main__':
    main()
