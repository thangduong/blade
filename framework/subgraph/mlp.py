import copy
import tensorflow as tf
import numpy as np
from framework.subgraph.core import dropout
from framework.utils.graph import initializations
from framework.utils.graph import variables
from framework.utils.graph import activations


def fully_connected_network(in_nodes, layer_sizes, layer_residual=None, layer_activations='relu', dropout_keep_probs=1.0, in_vars=None, name='fcn'):
    """
    Fully Connected Feed Forward Neural Network.

    This function simply forwards to fully_connected_layer.
    @type in_nodes: list
    @param in_nodes: a list if nodes from the previous layer
    @param layer_sizes: a list of numbers that are sizes of the layers in this subgraph
    @param layer_residual: an array of 2 element arrays that determine the source and target of the residual
    @param layer_activations: a list of activation functions (actual functiosn or name of the function).  this list should be equal
     in size to layer_sizes or it should be None or it should be a single item.  when it's a single item, then all
     layers have this same activation function.
    @param in_vars: for variable tying.
    @param name: name of this subgraph
    @return:
       out_nodes, out_vars
       (standard subgraph output)
    """

    layer_sizes = copy.copy(layer_sizes)
    if not isinstance(layer_activations, list):
        layer_activations = [layer_activations] * len(layer_sizes)
    if not isinstance(dropout_keep_probs, list):
        dropout_keep_probs = [dropout_keep_probs] * len(layer_sizes)
    if not isinstance(layer_residual, list):
        layer_residual = [layer_residual] * len(layer_sizes)

    # implement multiple fully connected pieces for the first layer
    # this is in case we have multiple inputs.
    #
    with tf.variable_scope(name):
        out_nodes = in_nodes
        out_vars = {}
        res_src = None
        for i, (layer_size, layer_activation, dropout_keep_prob, layer_index, residual) in enumerate(zip(layer_sizes, layer_activations, dropout_keep_probs, range(len(layer_sizes)), layer_residual)):
            # check if we are at a residual source

            out_nodes, layer_vars = fully_connected_layer(out_nodes, layer_size, activation=layer_activation, in_vars=in_vars, name="layer_%d"%layer_index)
            out_vars.update(layer_vars)
    return out_nodes, out_vars


def fully_connected_layer(in_nodes, layer_size,
                    activation='relu', bias=True,
                    weights_init='truncated_normal', bias_init='truncated_normal',
                    regularizer=None, weight_decay=0.000, trainable=True,
                    restore=True, in_vars={}, dropout_keep_prob=1.0,
                    name="fc"):
    """
    Single Fully Connected Feed Forward NN Layer

    @type in_nodes: list
    @param in_nodes: a list if nodes from the previous layer
    @param layer_size: size of this layer
    @param activation: activation function
    @param bias: whether or not to add bias
    @param weights_init: w*x+b, this is how w is initialized
    @param bias_init: this is how b is initialized
    @param regularizer: whether or not to use regularization and what regularization to use.
    @param weight_decay: parameter for regularization
    @param trainable: whether this subgraph is trainable
    @param restore: whether to save this variable when saving/restoring.  this currently doesn't work.
    @param in_vars: currently not implemented, but this is here for parameter tying.
    @param dropout_keep_prob: if set to 1.0, then no droup out. otherwise, this is the droupout keep prob
    @param name: name of this subgraph
    @return:
    """

    in_node_index = 0
    matmul_node_list = []
    out_variables = {}

    with tf.variable_scope(name+"_"):
        for in_node in in_nodes:
            input_shape = in_node.get_shape().as_list()
#      assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
            n_inputs = int(np.prod(input_shape[1:]))
            if len(input_shape) > 2:
                in_node = tf.reshape(in_node, [-1, n_inputs])
                # TODO: log when mismatch

            # w * x + b
            # this is the w part
            W_shape = [n_inputs, layer_size]
            W_init = weights_init
            if isinstance(weights_init, str):
                W_init = initializations.get(weights_init)(W_shape, stddev=0.35)
            W_regul = None
            if regularizer:
                W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
            W_name = 'W_' + str(in_node_index)
            W = variables.variable(W_name, shape=W_shape, regularizer=W_regul,
                            initializer=W_init, trainable=trainable,
                            restore=restore)
            out_variables[W_name] = W
            matmul_node_list.append(tf.matmul(in_node,W))
            in_node_index += 1

        out_node = tf.add_n(matmul_node_list)
        if (dropout_keep_prob < 1):
            [out_node], _ = dropout([out_node], dropout_keep_prob)

        # this is the b part
        b_shape = [layer_size]
        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)(b_shape, stddev=0.35)
            b = variables.variable('b', shape=b_shape, initializer=bias_init,
                            trainable=trainable, restore=restore)
            out_variables['b'] = b
            out_node = tf.nn.bias_add(out_node, b)

        if isinstance(activation, str):
            out_node = activations.get(activation)(out_node, name='activation')
        elif hasattr(activation, '__call__'):
            out_node = activation(out_node, name='activation')
        elif activation is None:
            None
        else:
            raise ValueError("Invalid Activation.")

    out_node = tf.identity(out_node, name=name)
    return [out_node], out_variables
