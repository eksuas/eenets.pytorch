from keras import losses
from keras import backend as K
from keras.models import Model
from keras.callbacks import Callback

import figures
import numpy as np
import tensorflow as tf

class CustomCallback(Callback):
    def __init__(self, net, name, (x_test, y_test)):
        self.num_layer = net.num_ee
        self.T = net.T
        self.filename = name
        self.x_test = x_test
        self.y_test = y_test
        self.c = net.rates
        self.flops = net.flops

    def on_train_begin(self, logs={}):
        self.h = [0 for i in xrange(self.num_layer+1)]
        self.l = [0 for i in xrange(self.num_layer+1)]
        self.L = [0 for i in xrange(self.num_layer+1)]
        self.accHist  = []
        self.costHist = []
        self.f = open(self.filename+'/losses.txt', 'w')
        for i in xrange(self.num_layer+1):
            self.f.write('       l%d'%i)
        for i in xrange(self.num_layer+1):
            self.f.write('       h%d'%i)
        for i in xrange(self.num_layer+1):
            self.f.write('       c%d'%i)
        for i in xrange(self.num_layer+1):
            self.f.write('       L%d'%i)
        self.f.write('\n')

    def on_epoch_begin(self, epoch, logs={}):
        self.acc  = np.full((self.y_test.shape[0]), -1)
        self.cost = np.full((self.y_test.shape[0]), -1)

    def on_epoch_end(self, epoch, logs={}):
        y_true = np.array([np.argmax(x) for x in self.y_test])
        target = self.y_test.astype('float32')

        for i in xrange(self.num_layer):
            conf = Model(inputs=self.model.input, outputs=self.model.get_layer(
                        'conf'+str(i)).output).predict(self.x_test).flatten()
            out  = Model(inputs=self.model.input, outputs=self.model.get_layer(
                        'out'+str(i)).output).predict(self.x_test)
            y_pred = np.array([np.argmax(x) for x in out])

            curr_acc = (conf >= self.T)*(y_pred == y_true) + (conf < self.T)*-1
            self.acc = (self.acc < 0)*curr_acc + (self.acc >= 0)*self.acc

            curr_cost = (conf >= self.T)*self.c[i] + (conf < self.T)*-1
            self.cost = (self.cost < 0)*curr_cost + (self.cost >= 0)*self.cost

            output = out.astype('float32')
            loss   = np.mean(K.eval( losses.categorical_crossentropy(K.constant(target), K.constant(output)) ))
            
            self.l[i] = loss
            self.h[i] = np.mean(conf)
    

        out = Model(inputs=self.model.input, outputs=self.model.get_layer(
                        'out'+str(self.num_layer)).output).predict(self.x_test)
        y_pred = [np.argmax(x) for x in out]

        self.acc = (self.acc < 0)*(y_pred == y_true) + (self.acc >= 0)*self.acc
        self.cost = (self.cost < 0)*1 + (self.cost >= 0)*self.cost
        
        output = out.astype('float32')
        loss   = np.mean(K.eval( losses.categorical_crossentropy(K.constant(target), K.constant(output)) ))
        
        self.l[self.num_layer] = loss      
        self.h[self.num_layer] = 1.0
        self.L[self.num_layer] = self.l[self.num_layer] + 2*self.c[self.num_layer]
        for i in range(self.num_layer-1,-1,-1):
            self.L[i] = 2*self.c[i] + self.h[i] * self.l[i] + (1-self.h[i]) * self.L[i+1]

        for i in xrange(self.num_layer+1):
            self.f.write('%9.3f'%self.l[i])
        for i in xrange(self.num_layer+1):
            self.f.write('%9.3f'%self.h[i])
        for i in xrange(self.num_layer+1):
            self.f.write('%9.3f'%self.c[i])
        for i in xrange(self.num_layer+1):
            self.f.write('%9.3f'%self.L[i])
        self.f.write('\n')

        self.accHist.append(np.mean(self.acc))
        self.costHist.append(np.mean(self.cost))
        print "accuracy: ", self.accHist[-1]
        print "cost:     ", self.costHist[-1]

    def on_train_end(self, logs={}):
        #figures.displayDetails(self.confHist, self.lossHist, self.LossHist, self.accHist, self.num_layer, self.filename)
        self.f.close()
