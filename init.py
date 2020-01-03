"""
initializer methods are defined in this code.
"""
import os
import csv
import argparse
import inspect
import six
import torch
#pylint: disable=W0611
from torch import nn
from torch.optim import Adam
from torch.optim import SGD
from custom_eenet import CustomEENet
from custom_eenet import eenet8
from eenet import EENet
from eenet import eenet18, eenet34, eenet50, eenet101, eenet152
from eenet import eenet20, eenet32, eenet44, eenet56, eenet110
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from resnet import set_complexity
from flops_counter import flops_to_string, params_to_string
#pylint: enable=W0611

def initializer():
    """initializer of the program.

    This parses and extracts all training and testing settings.
    """
    #pylint: disable=C0326, C0330
    parser = argparse.ArgumentParser(description='PyTorch Early-Exit Convolutional Neural Nets')
    parser.add_argument('--batch-size',   type=int,   default=32, metavar='N',
                                          help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch',   type=int,   default=1, metavar='N',
                                          help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs',       type=int,   default=150, metavar='N',
                                          help='number of epochs to train (default: 150)')
    parser.add_argument('--lr',           type=float, default=0.001, metavar='N',
                                          help='learning rate (default: 0.001)')
    parser.add_argument('--adaptive-lr',  action='store_true', default=False,
                                          help='adjust the learning rate')
    parser.add_argument('--momentum',     type=float, default=0.9, metavar='N',
                                          help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='N',
                                          help='weight decay for optimizers (default: 0.0001)')
    parser.add_argument('--no-cuda',      action='store_true', default=False,
                                          help='disable CUDA training')
    parser.add_argument('--multi-gpu',    action='store_true', default=False,
                                          help='enable multi-gpu training')
    parser.add_argument('--seed',         type=int,   default=1, metavar='N',
                                          help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int,   default=1, metavar='N',
                                          help='how many epochs to wait before logging training '+\
                                           'status (default: 1)')
    parser.add_argument('--no-save-model',action='store_true', default=False,
                                          help='do not save the current model')
    parser.add_argument('--load-model',   type=str,   default=None, metavar='S',
                                          help='the path for loading and evaluating model')
    parser.add_argument('--testing',      action='store_true', default=False,
                                          help='activate testing to loaded model')
    parser.add_argument('--filters',      type=int,   default=2, metavar='N',
                                          help='initial filters of custom eenet-8 (default: 2)')
    parser.add_argument('--lambda-coef',  type=float, default=1.0, metavar='N',
                                          help='lambda to arrange the balance between accuracy '+\
                                           'and cost (default: 1.0)')
    parser.add_argument('--num-ee',       type=int,   default=2, metavar='N',
                                          help='the number of early exit blocks (default: 2)')
    parser.add_argument('--dataset',      type=str,   default='cifar10',
                                          choices=['mnist','cifar10','svhn','imagenet',
                                           'tiny-imagenet'],
                                          help='dataset to be evaluated (default: cifar10)')
    parser.add_argument('--loss-func',    type=str,   default='v2', choices=['v1','v2','v3'],
                                          help='loss function (default: v2)')
    parser.add_argument('--optimizer',    type=str,   default='Adam', choices=['SGD','Adam'],
                                          help='optimizer (default: Adam)')
    parser.add_argument('--distribution', type=str,   default='fine',
                                          choices=['gold_ratio', 'pareto', 'fine', 'linear'],
                                          help='distribution method of exit blocks (default: fine)')
    parser.add_argument('--exit-type',    type=str,   default='pool', choices=['plain', 'pool',
                                           'bnpool'],
                                          help='Exit block type.')
    parser.add_argument('--model',        type=str,   default='eenet20',
                                          choices=['eenet8',
                                           'eenet18', 'eenet34', 'eenet50', 'eenet101', 'eenet152',
                                           'eenet20', 'eenet32', 'eenet44', 'eenet56',  'eenet110',
                                           'resnet18','resnet34','resnet50','resnet101','resnet152',
                                           'resnet20','resnet32','resnet44','resnet56', 'resnet110'
                                          ],
                                          help='model to be evaluated (default: eenet20)')
    parser.add_argument('--device',       help=argparse.SUPPRESS)
    parser.add_argument('--start-epoch',  help=argparse.SUPPRESS)
    parser.add_argument('--recorder',     help=argparse.SUPPRESS)
    parser.add_argument('--results-dir',  help=argparse.SUPPRESS)
    parser.add_argument('--models-dir',   help=argparse.SUPPRESS)
    parser.add_argument('--hist-file',    help=argparse.SUPPRESS)
    parser.add_argument('--num-classes',  help=argparse.SUPPRESS, default=10)
    parser.add_argument('--input-shape',  help=argparse.SUPPRESS, default=(3, 32, 32))
    #pylint: enable=C0326, C0330
    args = parser.parse_args()

    if args.dataset == 'mnist':
        args.input_shape = (1, 28, 28)

    elif args.dataset == 'imagenet':
        args.num_classes = 1000
        args.input_shape = (3, 224, 224)

    elif args.dataset == 'tiny-imagenet':
        args.num_classes = 200
        args.input_shape = (3, 64, 64)

    if args.model == 'eenet8':
        args.num_ee = 2

    if args.model[:6] == "resnet":
        args.num_ee = 0

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if use_cuda else 'cpu')

    # model configurations
    kwargs = vars(args)
    if args.load_model is None:
        model_object = _get_object(args.model)
        model = model_object(**kwargs)

        # create model folder
        args.models_dir = '../models/'+args.dataset+'/'+args.model
        if args.num_ee > 0:
            args.models_dir += '/ee'+str(args.num_ee)+'_'+args.distribution
        if not os.path.exists(args.models_dir):
            os.makedirs(args.models_dir)

        # continue to broken training
        args.start_epoch = 1
        while os.path.exists(args.models_dir+'/model'+'.v'+str(args.start_epoch)+'.pt'):
            args.start_epoch += 1
        if args.start_epoch > 1:
            model = torch.load(args.models_dir+'/model'+'.v'+str(args.start_epoch-1)+'.pt')
            if not isinstance(model, (CustomEENet, EENet)):
                set_complexity(model)
    else:
        model = torch.load(args.load_model)
        if not isinstance(model, (CustomEENet, EENet)):
            set_complexity(model)
        else:
            args.num_ee = model.num_ee
            args.model = "eenet"+str(model.total_layers)
            args.exit_type = model.exit_type
            args.distribution = model.distribution

    # use multiple GPU
    if use_cuda and torch.cuda.device_count() > 1 and args.multi_gpu:
        model.set_multiple_gpus()
        #model = nn.DataParallel(model)

    model = model.to(args.device)

    # optimizer configurations
    optimizer_object = _get_object(args.optimizer)
    keys = kwargs.keys() & inspect.getfullargspec(optimizer_object).args
    optimizer_args = {k:kwargs[k] for k in keys}
    optimizer = optimizer_object(model.parameters(), **optimizer_args)

    # print cost of exit blocks
    total_flops, _ = model.complexity[-1]
    for i, (flops, params) in enumerate(model.complexity[:-1]):
        print('ee-block-{}: flops={}, params={}, cost-rate={:.2f}'
              .format(i, flops_to_string(flops), params_to_string(params), flops/total_flops))
    flops, params = model.complexity[-1]
    print('exit-block: flops={}, params={}, cost-rate={:.2f}'
          .format(flops_to_string(flops), params_to_string(params), flops/total_flops))

    # create result folder
    args.results_dir = '../results/'+args.dataset+'/'+args.model
    if args.num_ee > 0:
        args.results_dir += '/ee'+str(args.num_ee)+'_'+args.distribution
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    args.hist_file = open(args.results_dir+'/history.csv', 'a', newline='')
    args.recorder = csv.writer(args.hist_file, delimiter=',')
    if os.stat(args.results_dir+'/history.csv').st_size == 0:
        args.recorder.writerow(['acc', 'acc_sem',
                                'cost', 'cost_sem',
                                'epoch',
                                'flop', 'flop_sem',
                                'time', 'time_sem',
                                'train_loss', 'train_loss_sem',
                                'val_loss', 'val_loss_sem'])

    return model, optimizer, args


def _get_object(identifier):
    """Cbject getter.

    This creates instances of the command line arguments by getting related objects.
    """
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier
