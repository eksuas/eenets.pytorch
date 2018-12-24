from keras import losses
from keras import backend as K
from keras import optimizers
from keras.utils import np_utils
from keras.utils import plot_model
from keras.utils import print_summary
from keras.models import model_from_json
from keras.models import Model
from keras.models import load_model
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.callbacks import Callback, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from callbacks import CustomCallback

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import nn
import figures
import json
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Train our network on MNIST, CIFAR10, CIFAR100 or SVHN',       \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',      type=str,   default='mnist', choices=['svhn', 'cifar10', 'cifar100',  \
                                      'mnist'],   help='Choose between MNIST and CIFAR.')
parser.add_argument('--arch',         type=str,   default='resnet18',                                       \
                                      help='Model architecture.')
parser.add_argument('--epoch',        type=int,   default=1,                                                \
                                      help='Number of epoch to train.')
parser.add_argument('--batch_size',   type=int,   default=32,                                               \
                                      help='Batch size.')
parser.add_argument('--loss',         type=str,   default='combined', choices=['single','combined'],        \
                                      help='Loss function')
parser.add_argument('--optimizer',    type=str,   default='adam',                                           \
                                      help='Optimizer function.', choices=['adam', 'sgd'])
parser.add_argument('--data_aug',     type=bool,  default=False,                                            \
                                      help='If data augmentation is performed.')
parser.add_argument('--save_name',    type=str,   default='',                                               \
                                      help='Model name to save arch, parameters, results etc.')
parser.add_argument('--load_model',   type=str,   default=None,                                             \
                                      help='File to load model.')
parser.add_argument('--verbose',      type=int,   default=1, choices=[0, 1, 2],                             \
                                      help='Verbosity: 0(silent), 1(prog-bar), or 2(one-line).')
parser.add_argument('--T',            type=float, default=0.5,                                              \
                                      help='Threshold of laters\' exit gate.')
parser.add_argument('--nb_classes',   type=int,   default=10,                                               \
                                      help='The number of classes dataset has.')
parser.add_argument('--exit_block',   type=str,   default='plain', choices=['plain', 'pool', 'bnpool'],     \
                                      help='Exit block type.')
parser.add_argument('--figures',      type=bool,  default=False,                                            \
                                      help='If train accuracy and cost figures are created.')
parser.add_argument('--print_summary',type=bool,  default=False,                                            \
                                      help='Print the summary of architexture.')
parser.add_argument('--num_ee',       type=int,   default=0,                                                \
                                      help='The number of eary exit blocks.')
parser.add_argument('--distribution', type=str,   default='gold_ratio', choices=['gold_ratio', 'pareto',    \
                                      'fine', 'linear', 'quad'], help='Distribution method of the exits.')


arg = parser.parse_args()


def main():
    if arg.save_name == '':
        arg.save_name = 'snapshots/'+arg.arch+'_'+arg.exit_block
    
    # get dataset
    train_data, test_data, dimensions = configuration()     

    # Create saving folder
    if not os.path.exists(arg.save_name):
        os.makedirs(arg.save_name)

    # Compute the number of floating point operations
    np.set_printoptions(precision=2)
    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)

        # Load model architecture from file
        if arg.load_model:
            arg.save_name = arg.load_model
            model = model_from_json(json.load(open(arg.load_model + '/arch.json')))
            with open(arg.load_model + '/config.txt', 'r') as fconfig:
                num_ee = int(fconfig.readline())
                lines  = fconfig.readlines()
                flops  = map(int, lines[:num_ee+1])
                rates  = map(float, lines[num_ee+1:])
            net = nn.NN(model, num_ee, flops, rates, arg.T)
            compile_model(net, train_data, test_data)
            model.load_weights(arg.load_model + '/weights.h5')

        else: 
            version = 1
            while os.path.exists(arg.save_name + '/v' + str(version)):
                version += 1    
        
            arg.save_name += '/v' + str(version)
            os.makedirs(arg.save_name)
            net = nn.buildModel(dimensions, arg)
            compile_model(net, train_data, test_data)
            train(net, train_data, test_data)
            # Save the model
            with open(arg.save_name + '/arch.json', 'w') as archfile:
                json.dump(net.model.to_json(), archfile)
            net.model.save_weights(arg.save_name + '/weights.h5') 
            with open(arg.save_name + '/config.txt', 'w') as fconfig:
                fconfig.write('{:d}\n'.format(net.num_ee))
                for flop in net.flops:
                    fconfig.write('{:d}\n'.format(flop))
                for rate in net.rates:
                    fconfig.write('{:.6f}\n'.format(rate))      

        # Predict model accuracy and cost
        acc, cost, _ = predict(net, test_data)
        print("\nT: %.2f --> acc: %.6f cost: %.6f" % (net.T, acc, cost))

    # For visualization of the model
    for idx, flop in enumerate(net.flops):
        print("out%d %10d %.2f%%" % (idx, flop, net.rates[idx]))
    plot_model(net.model, to_file=arg.save_name + '/vis.png', show_layer_names=True, show_shapes=True) 
    if arg.print_summary:
        print_summary(net.model)


def configuration():
    # Input image dimensions and channel
    if arg.dataset == 'cifar100':
        arg.nb_classes = 100

    dimensions = (3, 32, 32)
    if arg.dataset == 'mnist':
        dimensions = (1, 28, 28)

    # The data, shuffled and split between train and test sets:
    print 'Using', arg.dataset, 'dataset'

    if arg.dataset == 'svhn':
        train_data = sio.loadmat('svhn/train_32x32.mat')
        test_data  = sio.loadmat('svhn/test_32x32.mat')
        x_train, y_train = train_data['X'], train_data['y']
        x_test, y_test = test_data['X'], test_data['y']

        x_train = np.rollaxis(x_train, 3, 0)
        x_test = np.rollaxis(x_test, 3, 0)
        y_train[y_train == 10] = 0
        y_test[y_test == 10] = 0

    else:
        (x_train, y_train), (x_test, y_test) = eval(arg.dataset).load_data()

    # For mnist dataset
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=3)
        y_train = np.expand_dims(y_train, axis=1)
        x_test = np.expand_dims(x_test, axis=3)
        y_test = np.expand_dims(y_test, axis=1)

    # Convert class vectors to binary class matrices.
    y_train = np_utils.to_categorical(y_train, arg.nb_classes)
    y_test  = np_utils.to_categorical(y_test, arg.nb_classes)

    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')

    # Subtract mean and normalize
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image
    x_train /= 128.
    x_test /= 128.

    return (x_train, y_train), (x_test, y_test), dimensions


def compile_model(net, (x_train, y_train), (x_test, y_test)):

    def custom_loss(y_true, y_pred):
        c_loss = [0]*(net.num_ee)+ [tf.reshape(tf.convert_to_tensor(net.rates[net.num_ee]), [-1, 1])]
        y_pred = [0]*(net.num_ee) + [net.model.get_layer("out"+str(net.num_ee)).output]        
        loss = losses.categorical_crossentropy(y_true, y_pred[net.num_ee]) + c_loss[net.num_ee]
        for i in range(net.num_ee-1,-1,-1):
            out  = net.model.get_layer("out"+str(i)).output
            conf = net.model.get_layer("conf"+str(i)).output
            cost = tf.reshape(tf.convert_to_tensor(net.rates[i]), [-1, 1])

            c_loss[i] = conf*cost + (1-conf)*c_loss[i+1]
            y_pred[i] = conf*out  + (1-conf)*y_pred[i+1]    
            loss += losses.categorical_crossentropy(y_true, y_pred[i]) + c_loss[i]
        return loss

    loss = [custom_loss] * (2*net.num_ee+1)
    net.model.compile(optimizer=arg.optimizer, loss=loss, metrics=['accuracy'])


def predict(net, (x_test, y_test)):
    res  = np.full((y_test.shape[0]), -1)
    acc  = np.full((y_test.shape[0]), -1)
    exit = np.full((y_test.shape[0]), -1)
    hout = np.full((y_test.shape[0]), -1)

    confile = open(arg.save_name + '/confidence.txt', 'w')

    y_true = np.array([np.argmax(x) for x in y_test])
    for l in xrange(net.num_ee):
        conf = Model(inputs=net.model.input, outputs=net.model.get_layer('conf'+str(l)).output)  \
                    .predict(x_test).flatten()
        out  = Model(inputs=net.model.input, outputs=net.model.get_layer('out'+str(l)).output)   \
                    .predict(x_test)

        confile.write('Confidence values for exit{:d}\n'.format(l))
        for h in conf: 
            if h < net.T:       
                confile.write('{:.6f}\n'.format(h))

        y_pred = np.array([np.argmax(x) for x in out])
        curr_res  = (conf < net.T)*-1 + (conf >= net.T)*y_pred
        curr_acc  = (conf < net.T)*-1 + (conf >= net.T)*(y_pred == y_true)
        curr_exit = (conf < net.T)*-1 + (conf >= net.T)*l
        conf      = (conf < net.T)*-1 + (conf >= net.T)*conf

        res  = (res  >= 0)*res  + (res  < 0)*curr_res
        acc  = (acc  >= 0)*acc  + (acc  < 0)*curr_acc
        exit = (exit >= 0)*exit + (exit < 0)*curr_exit
        hout = (hout >= 0)*hout + (hout < 0)*conf

    out = Model(inputs=net.model.input, outputs=net.model.get_layer('out'+str(net.num_ee)).output) \
                .predict(x_test)
    y_pred = [np.argmax(x) for x in out]
    res  = (res  >= 0)*res  + (res  < 0)*y_pred
    acc  = (acc  >= 0)*acc  + (acc  < 0)*(y_pred == y_true)
    exit = (exit >= 0)*exit + (exit < 0)*net.num_ee
    hout = (hout >= 0)*hout + (hout < 0)*1
    
    with open(arg.save_name + '/prediction.txt', 'w') as resfile:
        cost = 0.0
        for i in xrange(len(res)):
            if acc[i]:
                resfile.write('%5d. true       conf:%.4f \texit%d\n'%(i, hout[i], exit[i]))
            else:
                resfile.write('%5d. (P:%d->T:%d) conf:%.4f \texit%d\n'%(i, res[i], y_true[i], hout[i], exit[i]))
            cost += net.rates[exit[i]]
        cost /= len(res) 
    
    return np.mean(acc), cost, res


def train(net, (x_train, y_train), (x_test, y_test)):
    # Callbacks configuration
    lr_reducer    = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=20)
    csv_logger    = CSVLogger(arg.save_name + '/logger.csv')
    layer_output  = CustomCallback(net, arg.save_name, (x_test, y_test))   
    
    callbacks=[lr_reducer, early_stopper, csv_logger]
    if arg.figures:
        callbacks.append(layer_output)

    train_data = {'out'+str(net.num_ee) : y_train}
    test_data  = {'out'+str(net.num_ee) : y_test}
    for i in xrange(net.num_ee):
        train_data['conf'+str(i)] = y_train
        train_data['out'+str(i)]  = y_train
        test_data['conf'+str(i)] = y_test
        test_data['out'+str(i)]  = y_test

    if not arg.data_aug:
        print('Not using data augmentation.')
        history = net.model.fit(x_train, train_data,
                            batch_size=arg.batch_size,
                            epochs=arg.epoch,
                            validation_data=(x_test, test_data),
                            shuffle=True,
                            verbose=arg.verbose,
                            callbacks=callbacks)

    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images


        def batch_generator(datagen, x_train, y_train, num_outputs):
            gen = datagen.flow(x_train, y_train, batch_size=arg.batch_size, seed=666)
            while True: # keras required all generators to be infinite
                data = gen.next()
                yield data[0], [data[1]] * num_outputs  

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        # datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        history = net.model.fit_generator(batch_generator(datagen, x_train, y_train, (2*net.num_ee+1)),
                            epochs=arg.epoch,
                            validation_data=(x_test, test_data),
                            verbose=arg.verbose, 
                            callbacks=callbacks,
                            max_queue_size=100,
                            steps_per_epoch=x_train.shape[0] // arg.batch_size)
    
    figures.display(history, net.num_ee, arg.save_name)


if __name__ == '__main__':
    main()

