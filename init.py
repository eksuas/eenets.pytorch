import argparse
import inspect
import torch
import six

from torch.optim import Adam
from torch.optim import SGD
from custom_eenet import *
from eenet import *
from resnet import *

def initializer():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',   type=int,   default=256, metavar='N',
                                          help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch',   type=int,   default=1, metavar='N',
                                          help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs',       type=int,   default=10, metavar='N',
                                          help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',           type=float, default=0.1, metavar='LR',
                                          help='learning rate (default: 0.1)')
    parser.add_argument('--momentum',     type=float, default=0.9, metavar='M',
                                          help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='M',
                                          help='weight decay for optimizers (default: 0.0001)')
    parser.add_argument('--no-cuda',      action='store_true', default=False,
                                          help='disables CUDA training')
    parser.add_argument('--seed',         type=int,   default=1, metavar='S',
                                          help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int,   default=10, metavar='N',
                                          help='how many batches to wait before logging training status')
    parser.add_argument('--save-model',   action='store_true', default=False,
                                          help='save current model')
    parser.add_argument('--load-model',   type=str, default='',
                                          help='the path for loading and evaluating model')
    parser.add_argument('--filters',      type=int,   default=2,
                                          help='initial filter number of basic eenets (default: 2)')
    parser.add_argument('--lamb',         type=float, default=1.0,
                                          help='lambda to arrange the balance between accuracy and cost \
                                          (default: 1.0)')
    parser.add_argument('--num-ee',       type=int,   default=2,
                                          help='the number of early exit blocks (default: 3)')
    parser.add_argument('--filename',     type=str,   default='modelChart',
                                          help='the filename of plots (default: modelChart)')
    parser.add_argument('--dataset',      type=str,   default='cifar10',
                                          choices=['mnist','cifar10','svhn','imagenet'],
                                          help='dataset to be evaluted (default: cifar10)')
    parser.add_argument('--num-classes',  type=int,   default=10,
                                          help='the number of classes in the dataset (default: 10)')
    parser.add_argument('--optimizer',    type=str,   default='SGD', choices=['SGD','Adam'],
                                          help='optimizer (default: SGD)')
    parser.add_argument('--input-shape',  type=tuple, default=(3, 32, 32),
                                          help='the shape of dataset (default: (3, 32, 32))')
    parser.add_argument('--distribution', type=str,   default='fine',
                                          choices=['gold_ratio', 'pareto', 'fine', 'linear'],
                                          help='distribution method of exit blocks (default: fine)')
    parser.add_argument('--model',        type=str,   default='eenet20',
                                          choices=['eenet8',
                                           'eenet18', 'eenet34', 'eenet50', 'eenet101', 'eenet152',
                                           'eenet20', 'eenet32', 'eenet44', 'eenet56',  'eenet110',
                                           'resnet18','resnet34','resnet50','resnet101','resnet152',
                                           'resnet20','resnet32','resnet44','resnet56', 'resnet110'
                                          ],
                                          help='model to be evaluated (default: eenet20)')

    args = parser.parse_args()

    if (args.dataset == 'mnist'):
        args.input_shape = (1, 28, 28)

    elif (args.dataset == 'imagenet'):
        args.num_classes = 1000
        args.input_shape = (3, 224, 224)

    torch.manual_seed(args.seed)
    Model = _get_object(args.model)

    # optimizer configurations
    Optimizer = _get_object(args.optimizer)
    argspec = inspect.getargspec(Optimizer)
    optimizer_args = {}
    kwargs = vars(args)
    for key, value in kwargs.items():
        if (key in argspec.args):
            optimizer_args[key] = value

    return Model, Optimizer, args, optimizer_args

def _get_object(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier
