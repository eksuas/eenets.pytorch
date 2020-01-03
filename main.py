"""
high level support for doing this and that.
"""
from __future__ import print_function
import time
import csv
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from init import initializer
from eenet import EENet
from custom_eenet import CustomEENet
import utils


def main():
    """Main function of the program.

    The function loads the dataset and calls training and validation functions.
    """
    model, optimizer, args = initializer()
    train_loader, test_loader = utils.load_dataset(args)

    # disable training
    if args.testing:
        result = validate(args, model, test_loader)
        print('\nThe avg val_loss: {:.4f}, avg val_cost: {:.2f}%, avg val_acc: {:.2f}%\n'
              .format(result['val_loss'], result['cost'], result['acc']))
        examine(args, model, test_loader)
        return

    for epoch in range(args.start_epoch, args.epochs + 1):
        exit_id = utils.get_active_exit(args, epoch)
        print('{:3d}: Exit-ID:{:2d} '.format(epoch, exit_id), end='')

        # use adaptive learning rate
        if args.adaptive_lr:
            utils.adaptive_learning_rate(args, optimizer, epoch)

        result = {'epoch':epoch}
        result.update(train(args, exit_id, model, train_loader, optimizer))

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
    # utils.plot_history(args)


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


def train(args, exid_id, model, train_loader, optimizer):
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
            cost.append(torch.tensor(1.0).to(args.device))

            # loss v1
            if args.loss_func == 'v1':
                cum_pred = pred[args.num_ee]
                cum_cost = cost[args.num_ee]
                for i in range(args.num_ee-1, exid_id-1, -1):
                    cum_pred = conf[i] * pred[i] + (1-conf[i]) * cum_pred
                    cum_cost = conf[i] * cost[i] + (1-conf[i]) * cum_cost
                loss = F.nll_loss(cum_pred.log(), target)
                cum_loss = loss + args.lambda_coef * cum_cost.mean()

            # loss v2
            # Calculate cumulative prediction and cost during training
            if args.loss_func == 'v2':
                cum_pred = [None] * args.num_ee + [pred[args.num_ee]]
                cum_cost = [None] * args.num_ee + [cost[args.num_ee]]
                loss = F.nll_loss(cum_pred[-1].log(), target)
                cum_loss = loss + args.lambda_coef*cum_cost[-1].mean()
                for i in range(args.num_ee-1, exid_id-1, -1):
                    cum_pred[i] = conf[i] * pred[i] + (1-conf[i]) * cum_pred[i+1]
                    cum_cost[i] = conf[i] * cost[i] + (1-conf[i]) * cum_cost[i+1]
                    loss = F.nll_loss(cum_pred[i].log(), target)
                    cum_loss += loss + args.lambda_coef * cum_cost[i].mean()

            # loss v3
            if args.loss_func == 'v3':
                conf_sum = 0
                for i in range(len(conf)):
                    conf_sum += conf[i].mean()
                norm_conf = [conf[i].mean() / conf_sum for i in range(len(conf))]

                loss = F.nll_loss(pred[args.num_ee].log(), target)
                cum_loss = norm_conf[args.num_ee] * (loss + args.lambda_coef * cost[args.num_ee].mean())
                for i in range(args.num_ee-1, -1, -1):
                    loss = F.nll_loss(pred[i].log(), target)
                    cum_loss += norm_conf[i] * (loss + args.lambda_coef * cost[i].mean())

        # training settings for other models
        else:
            pred = model(data)
            loss = F.cross_entropy(pred, target)

        losses.append(float(cum_loss))
        cum_loss.backward()
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


if __name__ == '__main__':
    main()
