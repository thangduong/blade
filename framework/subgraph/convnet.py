import tensorflow as tf
import framework.utils.graph.utils as utils
import framework.utils.graph.initializations as initializations
import framework.utils.graph.variables as variables
import framework.utils.graph.activations as activations

def conv_2d(in_nodes, nb_filter, filter_size, strides=1, padding='same',
            activation='relu', bias=True, weights_init='truncated_normal',
            bias_init='truncated_normal', regularizer=None, weight_decay=0.001,
            trainable=True, restore=True, reuse=False,
            name="conv_2d"):
    """ Convolution 2D.
    Input:
        4-D Tensor [batch, height, width, in_channels].
    Output:
        4-D Tensor [batch, new height, new width, nb_filter].
    Arguments:
        incoming: `Tensor`. Incoming 4-D Tensor.
        nb_filter: `int`. The number of convolutional filters.
        filter_size: `int` or `list of int`. Size of filters.
        strides: 'int` or list of `int`. Strides of conv operation.
            Default: [1 1 1 1].
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'Conv2D'.
    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        b: `Variable`. Variable representing biases.
    """
    input_shape = utils.get_incoming_shape(in_nodes[0])
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
    filter_size = utils.autoformat_filter_conv2d(filter_size, input_shape[-1], nb_filter)
    strides = utils.autoformat_kernel_2d(strides)
    padding = utils.autoformat_padding(padding)
    out_vars = {}

    with tf.variable_scope(name):
        # conv
        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = variables.variable('W', shape=filter_size, regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)
        out_vars['W'] = W
        out_node = tf.nn.conv2d(in_nodes[0], W, strides, padding, name='conv')

        # bias
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = variables.variable('b', shape=nb_filter, initializer=bias_init,
                            trainable=trainable, restore=restore)
            out_node = tf.add(out_node, b, name='bias')
            out_vars['b'] = b

        # activation
        if isinstance(activation, str):
            out_node = activations.get(activation)(out_node, name='activation')
        elif hasattr(activation, '__call__'):
            out_node = activation(out_node, name='activation')
        else:
            raise ValueError("Invalid Activation.")

    return [out_node], out_vars

def max_pool_2d(in_nodes, kernel_size, strides=None, padding='same', name="max_pool_2d"):
    kernel_size = utils.autoformat_kernel_2d(kernel_size)
    strides = utils.autoformat_kernel_2d(strides) if strides else kernel
    padding = utils.autoformat_padding(padding)
    out_node = tf.nn.max_pool(in_nodes[0], kernel_size, strides, padding, name=name)
    return [out_node], {}

def conv_1d(in_nodes, )