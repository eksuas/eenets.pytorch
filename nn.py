# from __future__ import division
import six
import operator
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers.merge import add
from keras.layers.merge import multiply 
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters,
                                   init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def exit_func(units, kernel_initializer, activation, name, bias_initializer="zeros"):
    """Define exit learnable function
    """
    def f(input):
        output = Dense( units=units, 
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer, 
                        activation=activation, 
                        name=name)(input)
        return output

    return f


def plain_exit_block(layer_id, num_outputs):
    """Exit blocks for confidence and softmax outputs
    """
    def f(input):
        shape = K.int_shape(input)
        # Classifier blocks for softmaxs and confidence values
        output = Flatten()(input)

        layer_out = exit_func(units=num_outputs, kernel_initializer="he_normal",
                                activation="softmax", name="out"+str(layer_id))(output)

        halting_score = exit_func(units=1, kernal_initializer="He_normal",
                                activation="sigmoid", name="conf"+str(layer_id))(output)
                                #bias_initializer=keras.initializer.Constant(value=0.5))(output)
        return layer_out, halting_score

    return f


def pool_exit_block(layer_id, num_outputs):
    """Exit blocks for confidence and softmax outputs
    """
    def f(input):
        shape = K.int_shape(input)
        # Classifier blocks for softmaxs and confidence values
        output = AveragePooling2D(pool_size=(shape[ROW_AXIS], shape[COL_AXIS]), strides=(1,1))(input)
        output = Flatten()(output)

        layer_out = exit_func(units=num_outputs, kernel_initializer="he_normal",
                              activation="softmax", name="out"+str(layer_id))(output)

        halting_score = exit_func(units=1, kernel_initializer="he_normal",
                              activation="sigmoid", name="conf"+str(layer_id))(output)
        return layer_out, halting_score

    return f


def bnpool_exit_block(layer_id, num_outputs):
    """Exit blocks for confidence and softmax outputs
    """
    def f(input):
        shape = K.int_shape(input)
        # Classifier blocks for softmaxs and confidence values
        output = _bn_relu(input)
        output = AveragePooling2D(pool_size=(shape[ROW_AXIS], shape[COL_AXIS]), strides=(1,1))(output)
        output = Flatten()(output)    
    
        layer_out = exit_func(units=num_outputs, kernel_initializer="he_normal",
                              activation="softmax", name="out"+str(layer_id))(output)

        halting_score = exit_func(units=1, kernel_initializer="he_normal",
                              activation="sigmoid", name="conf"+str(layer_id))(output)
        return layer_out, halting_score

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def getNumberOfOutLayer(model):
    res = 0
    for layer in model.layers:
        if layer.name[:3] == "out":
            res += 1
    return res


class NN:
    def __init__(self, model, num_ee, flops, rates, T):
        self.model = model
        self.num_ee = num_ee
        self.flops = flops
        self.rates = rates
        self.T = T

class NaiveResnetBuilder(object):
    @staticmethod
    def build(input_shape, block_fn, arg, repetitions):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops_path = []

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64

        outputs=[]
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=arg.nb_classes, kernel_initializer="he_normal",
                      activation="softmax", name='out'+str(arg.num_ee))(flatten1)

        flop_out = tf.profiler.profile(K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
        flops_path.append(flop_out.total_float_ops)
        flops_rate = [float(flop)/flops_path[-1] for flop in flops_path]

        outputs.append(dense)

        model = Model(inputs=input, outputs=outputs)
        return NN(model, 0, flops_path, flops_rate, arg.T)


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, block_fn, arg, rep):
        """Builds a 6n+2 ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)


        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops_path = []
        flops_gate = []

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=16, kernel_size=(3, 3), strides=(1, 1))(input)
        #pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
        #block = pool1

        filters = 16
        block = conv1        
        outputs=[]
        for i in xrange(3):
            block = _residual_block(block_fn, filters=filters, repetitions=rep, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=arg.nb_classes, kernel_initializer="he_normal",
                      activation="softmax", name='out'+str(arg.num_ee))(flatten1)

        
        flops_path = map(operator.add, flops_path, flops_gate)
        flop_out = tf.profiler.profile(K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
        flops_path.append(flop_out.total_float_ops-sum(flops_gate))
        flops_rate = [float(flop)/flops_path[-1] for flop in flops_path]

        outputs.append(dense)

        model = Model(inputs=input, outputs=outputs)
        #model = Model(inputs=input, outputs=dense)
        return NN(model, 0, flops_path, flops_rate, arg.T)


class NaiveEenetBuilder(object):
    @staticmethod
    def build(input_shape, block_fn, arg, repetitions):
        """Builds an EdaNet from naive ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)
        exit_block = _get_block(arg.exit_block+'_exit_block')

        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops_path = []
        flops_gate = []

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64

        outputs=[]
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            if i == len(repetitions)-1: break
            filters *= 2

            flop_in = tf.profiler.profile(K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
            
            out, conf = exit_block(layer_id=i, num_outputs=arg.nb_classes)(block)
            outputs.append(out)
            outputs.append(conf)
            
            flop_out = tf.profiler.profile(K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
            flop_of_gate = flop_out.total_float_ops - flop_in.total_float_ops

            flops_path.append(flop_out.total_float_ops + sum(flops_gate)) 
            flops_gate.append(flop_of_gate)


        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=arg.nb_classes, kernel_initializer="he_normal",
                      activation="softmax", name='out'+str(arg.num_ee))(flatten1)


        flop_out = tf.profiler.profile(K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
        flops_path.append(flop_out.total_float_ops + sum(flops_gate))
        flops_rate = [float(flop)/flops_path[-1] for flop in flops_path]

        outputs.append(dense)

        model = Model(inputs=input, outputs=outputs)
        return NN(model, len(flops_gate), flops_path, flops_rate, arg.T)


class EenetBuilder(object):
    @staticmethod
    def build(input_shape, block_fn, arg, stage, rep, init_filter):
        """Builds an EdaNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)
        exit_block = _get_block(arg.exit_block+'_exit_block')

        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops_path = []
        flops_gate = []

        #filters = 16
        filters = init_filter
        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=filters, kernel_size=(3, 3), strides=(1, 1))(input)
        #pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
        #block = pool1
        
        flop = tf.profiler.profile(K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
        blocks = [(conv1, flop.total_float_ops)]
        
        o_layers = []
        h_layers = []
        for i in xrange(stage):
            init_strides = (2,2)
            for j in range(rep):
                if i == 0: init_strides = (1, 1)
                block = block_fn(filters=filters,
                                init_strides=init_strides,
                                is_first_block_of_first_layer=(i == 0 and j == 0))(blocks[-1][0])
                flop = tf.profiler.profile(K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
                blocks.append((block, flop.total_float_ops))
                init_strides = (1,1)

            if i == 2: break
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=arg.nb_classes, kernel_initializer="he_normal",
                      activation="softmax", name='out'+str(arg.num_ee))(flatten1)


        flop = tf.profiler.profile(K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
        blocks.append((dense, flop.total_float_ops))    

        total_flop = flop.total_float_ops
        gold_rate = 1.61803398875
        flop_margin = 1.0 / (arg.num_ee+1)
        layer_margin = int(flop_margin * (stage*rep + 2))

        block_id = 0
        for e in xrange(arg.num_ee):
            if arg.distribution == "pareto":
                threshold = total_flop * (1 - (0.8**(e+1)))
            elif arg.distribution == "fine":
                threshold = total_flop * (1 - (0.95**(e+1)))
            elif arg.distribution == "linear":
                threshold = total_flop * flop_margin * (e+1)
            elif arg.distribution == "quad":
                _, threshold = blocks[layer_margin * (e+1)]
            else:
                threshold = total_flop * (gold_rate**(e - arg.num_ee))
            
            while block_id < len(blocks):
                block, flop = blocks[block_id]
                block_id += 1
                if flop >= threshold:

                    flop_in = tf.profiler.profile(K.get_session().graph, 
                                run_meta=run_meta, cmd='op', options=opts)

                    out, conf = exit_block(layer_id=e, num_outputs=arg.nb_classes)(block)
                    o_layers.append(out)
                    h_layers.append(conf)

                    flop_out = tf.profiler.profile(K.get_session().graph, 
                                run_meta=run_meta, cmd='op', options=opts)
                    flop_of_gate = flop_out.total_float_ops - flop_in.total_float_ops

                    flops_path.append(flop + sum(flops_gate))
                    flops_gate.append(flop_of_gate)
                    break

        flops_path.append(total_flop + sum(flops_gate))
        flops_rate = [float(flop)/flops_path[-1] for flop in flops_path]

        o_layers.append(dense)
        
        """    
        output = dense
        for i in range(arg.num_ee-1,-1,-1):
            p1 = multiply([h_layers[i], o_layers[i]])
            p2 = multiply([Lambda(lambda x: 1.0 - x)(h_layers[i]), output])
            output = add([p1, p2])
        """        

        model = Model(inputs=input, outputs=h_layers+o_layers)
        return NN(model, arg.num_ee, flops_path, flops_rate, arg.T)



def buildModel(input_shape, arg):
    if arg.arch == 'resnet18':
        return NaiveResnetBuilder.build(input_shape, basic_block, arg, [2, 2, 2, 2])
    elif arg.arch == 'resnet34':
        return NaiveResnetBuilder.build(input_shape, basic_block, arg, [3, 4, 6, 3])
    elif arg.arch == 'resnet50':
        return NaiveResnetBuilder.build(input_shape, bottleneck, arg, [3, 4, 6, 3])
    elif arg.arch == 'resnet101':
        return NaiveResnetBuilder.build(input_shape, bottleneck, arg, [3, 4, 23, 3])
    elif arg.arch == 'resnet152':
        return NaiveResnetBuilder.build(input_shape, bottleneck, arg, [3, 8, 36, 3])

    elif arg.arch == 'resnet20':
        return ResnetBuilder.build(input_shape, basic_block, arg, 3)
    elif arg.arch == 'resnet32':
        return ResnetBuilder.build(input_shape, basic_block, arg, 5)
    elif arg.arch == 'resnet44':
        return ResnetBuilder.build(input_shape, basic_block, arg, 7)
    elif arg.arch == 'resnet56':
        return ResnetBuilder.build(input_shape, basic_block, arg, 9)
    elif arg.arch == 'resnet110':
        return ResnetBuilder.build(input_shape, basic_block, arg, 18)

    elif arg.arch == 'eenet18':
        return NaiveEenetBuilder.build(input_shape, basic_block, arg, [2, 2, 2, 2])
    elif arg.arch == 'eenet34':
        return NaiveEenetBuilder.build(input_shape, basic_block, arg, [3, 4, 6, 3])
    elif arg.arch == 'eenet50':
        return NaiveEenetBuilder.build(input_shape, bottleneck, arg, [3, 4, 6, 3])
    elif arg.arch == 'eenet101':
        return NaiveEenetBuilder.build(input_shape, bottleneck, arg, [3, 4, 23, 3])
    elif arg.arch == 'eenet152':
        return NaiveEenetBuilder.build(input_shape, bottleneck, arg, [3, 8, 36, 3])

    elif arg.arch == 'eenet4':
        return EenetBuilder.build(input_shape, basic_block, arg, 1, 1, 2)
    elif arg.arch == 'eenet6':
        return EenetBuilder.build(input_shape, basic_block, arg, 2, 1, 2)
    elif arg.arch == 'eenet8':
        return EenetBuilder.build(input_shape, basic_block, arg, 3, 1, 4)
    elif arg.arch == 'eenet20':
        return EenetBuilder.build(input_shape, basic_block, arg, 3, 3, 16)
    elif arg.arch == 'eenet32':
        return EenetBuilder.build(input_shape, basic_block, arg, 3, 5, 16)
    elif arg.arch == 'eenet44':
        return EenetBuilder.build(input_shape, basic_block, arg, 3, 7, 16)
    elif arg.arch == 'eenet56':
        return EenetBuilder.build(input_shape, basic_block, arg, 3, 9, 16)
    elif arg.arch == 'eenet110':
        return EenetBuilder.build(input_shape, basic_block, arg, 3, 18, 16)

