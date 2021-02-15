"""
Edanur Demir
Training and validation of EENet
"""
from __future__ import print_function
import time
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy import stats
from init import initializer
from eenet import EENet
from custom_eenet import CustomEENet
import loss_functions
import utils


def main():
    """Main function of the program.

    The function loads the dataset and calls training and validation functions.
    """
    model, optimizer, args = initializer()
    train_loader, test_loader, exit_tags = utils.load_dataset(args)

    # disable training
    if args.testing:
        result = validate(args, model, test_loader)
        #print('\nThe avg val_loss: {:.4f}, avg val_cost: {:.2f}%, avg val_acc: {:.2f}%\n'
        #      .format(result['val_loss'], result['cost'], result['acc']))
        #examine(args, model, test_loader)
        return

    if args.two_stage:
        args.loss_func = "v0"

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('{:3d}:'.format(epoch), end='')

        # two-stage training uses the loss version-1 after training for 25 epochs
        if args.two_stage and epoch > 25:
            args.loss_func = "v1"

        # use adaptive learning rate
        if args.adaptive_lr:
            utils.adaptive_learning_rate(args, optimizer, epoch)

        result = {'epoch':epoch}
        result.update(train(args, model, train_loader, optimizer, exit_tags))

        # validate and keep history at each log interval
        if epoch % args.log_interval == 0:
            result.update(validate(args, model, test_loader))
            utils.save_history(args, result)

        # save model parameters
        if not args.no_save_model:
            utils.save_model(args, model, epoch)

    # print the best validation result
    best_epoch = utils.close_history(args)

    # save the model giving the best validation results as a final model
    if not args.no_save_model:
        utils.save_model(args, model, best_epoch, True)

    utils.plot_history(args)


def train(args, model, train_loader, optimizer, exit_tags):
    """train the model.

    Arguments are
    * args:         command line arguments entered by user.
    * model:        convolutional neural network model.
    * train_loader: train data loader.
    * optimizer:    optimize the model during training.
    * epoch:        epoch number.

    This trains the model and prints the results of each epochs.
    """
    # just printing for debug purposes
    if args.loss_func == 'v5':
        debug = [0 for i in range(args.num_ee+1)]
        for batch in exit_tags:
            for exit in batch:
                debug[exit] += 1
        print("<", end="")
        for exit in debug:
            print(str(exit) + ", ", end="")
        print(">")

    # initialize the lists
    losses = []
    pred_losses = []
    cost_losses = []
    model.train()

    # actual training starts
    for batch_id, (data, target) in enumerate(train_loader):
        # fetch the current batch data
        data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
        exit_tag = None
        if args.loss_func == 'v4':
            exit_tag = exit_tags[batch_id].to(args.device)

        optimizer.zero_grad()

        # training settings for EENet based models
        if isinstance(model, (CustomEENet, EENet)):
            pred, conf, cost = model(data)
            cost.append(torch.tensor(1.0).to(args.device))
            cum_loss, pred_loss, cost_loss = loss_functions.loss(args, exit_tag, pred,
                                                                 target, conf, cost)

        # training settings for other models (e.g. ResNet)
        else:
            pred = model(data)
            cum_loss = F.cross_entropy(pred, target)

        losses.append(float(cum_loss))
        pred_losses.append(float(pred_loss))
        cost_losses.append(float(cost_loss))
        cum_loss.backward()
        optimizer.step()


    # update the exit tags of inputs
    if args.loss_func == 'v4':
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
            batch_size = len(exit_tags[batch_id])
            pred, _, cost = model(data)

            exit_tags[batch_id] = loss_functions.update_exit_tags(args, batch_size,
                                                                  pred, target, cost)


    # print the training results of epoch
    result = {'train_loss': round(np.mean(losses), 4),
              'train_loss_sem': round(stats.sem(losses), 2),
              'pred_loss': round(np.mean(pred_losses), 4),
              'pred_loss_sem': round(stats.sem(pred_losses), 2),
              'cost_loss': round(np.mean(cost_losses), 4),
              'cost_loss_sem': round(stats.sem(cost_losses), 2)}

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
        for batch_id, (data, target) in enumerate(val_loader):
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
                flop, cost = 1.0, 1.0 #model.complexity[-1][0], 1.0
                exit_points = None

            # get the index of the max log-probability
            pred = pred.max(1, keepdim=True)[1]
            acc = pred.eq(target.view_as(pred)).sum().item()
            batch['acc'].append(acc*100.)
            batch['time'].append(elapsed_time)
            batch['cost'].append(cost*100.)
            batch['flop'].append(flop)
            batch['val_loss'].append(float(loss))

    utils.print_validation(args, batch, exit_points)

    result = {}
    for key, value in batch.items():
        result[key] = round(np.mean(value), 4)
        result[key+'_sem'] = round(stats.sem(value), 2)
    return result


def examine(args, model, val_loader):
    """examine the model output.

    Arguments are
    * args:         command line arguments entered by user.
    * model:        convolutional neural network model.
    * train_loader: train data loader.

    This examines the outputs of already trained model.
    """
    model.train()
    print(args.results_dir+'/pred_vs_conf.csv')
    experiment = open(args.results_dir+'/pred_vs_conf.csv', 'w', newline='')
    recorder = csv.writer(experiment, delimiter=',')
    recorder.writerow(['target',
                       'start_pred_seq',
                       'start_conf_seq',
                       'start_exit_seq',
                       'actual_pred',
                       'actual_conf',
                       'actual_exit'])
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
            preds, confs, costs = model(data)

            preds = [p.argmax().item() for p in preds]
            confs = [c.item() for c in confs]
            target = target.item()

            start_pred_seq = actual_pred = preds[-1]
            start_conf_seq = actual_conf = 1.0
            start_exit_seq = actual_exit = args.num_ee
            for i in range(args.num_ee):
                if confs[i] > 0.5:
                    actual_pred = preds[i]
                    actual_conf = confs[i]
                    actual_exit = i
                    break

            for i in range(args.num_ee-1, -1, -1):
                if preds[i] == start_pred_seq:
                    start_conf_seq = confs[i]
                    start_exit_seq = i
                else:
                    break

            recorder.writerow([target,
                               start_pred_seq,
                               start_conf_seq,
                               start_exit_seq,
                               actual_pred,
                               actual_conf,
                               actual_exit])
            experiment.flush()
    return


if __name__ == '__main__':
    main()
