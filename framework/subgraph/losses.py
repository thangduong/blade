"""
A bunch of nontrivial commonly used loss functions.

"""
import tensorflow as tf
import framework.utils.common as common_utils
import framework.subgraph.mlp as mlp
import numpy as np
from framework.utils.graph import initializations
from framework.utils.graph import variables

def contrastive(network, name='contrastive_loss', params=None):
    """
    Implement contrastive loss as given in LeCunn's paper.
      http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    @param network: input network that contains 3 nodes (left, right, label)
    @return:
    #label, margin, y_weight=1,l2_name=None, l2_norm_name=None, loss_name=None, params=None
    """
    assert(isinstance(network, list) and (len(network)==3), 'losses.contrastive: input must contain 3 nodes')
    with tf.variable_scope(name):
        margin = common_utils.get_dict_value(params, 'contrastive_loss_margin', 128)
        l2_name = common_utils.get_dict_value(params, 'l2_name', 'l2')
        l2_norm_name = common_utils.get_dict_value(params, 'l2_norm_name', 'l2_norm')
        left_feature = network[0]
        right_feature = network[1]
        label = network[2]
        one = tf.constant(1.0, dtype=tf.float32)
        zero = tf.constant(0.0, dtype=tf.float32)
        margin_constant = tf.constant(margin, dtype=tf.float32)
        label_sum = tf.reduce_sum(label)
        Y = tf.cond(label_sum>0, lambda: tf.mul(tf.div(tf.cast(tf.size(label),dtype=tf.float32), label_sum), label), lambda: label)
        one_minus_y = tf.sub(one, Y, name='one_minus_y')
        N = tf.constant(1 / float(right_feature.get_shape().as_list()[1]))
        Dw2 = tf.reduce_sum(tf.square(tf.sub(left_feature, right_feature)), 1, name=l2_name)
        right_term = tf.mul(Y, tf.square(tf.maximum(zero, tf.sub(margin_constant, tf.mul(N, tf.sqrt(Dw2, name=l2_norm_name))))), name='left_term')
        left_term = tf.mul(one_minus_y, tf.mul(tf.square(N), Dw2), name='right_term')
        loss = tf.mul(tf.constant(0.5), tf.reduce_mean(tf.add(right_term, left_term)), name=name)
    return [loss], {}


def lambda_ranknet_pairwise(network, params=None):
    None


def triplet(network, params=None):
    """
    Weinberger's LMNNC labeled triplet harmonic loss by Google's group.
    @param network:
    @param params:
    @return:
    """
    None


def class_accuracy_max(network_answers, labels, name="accuracy", compute_prediction=True):
    """
    Compute the accuracy based on max
    @param network_answers: 
    @param labels: 
    @param name: 
    @compute_prediction: if true, network_answers = softmax output, if false, network_answers = actual prediction class #
    @return: 
    """
    with tf.variable_scope(name+"_"):
        if compute_prediction:
            prediction = tf.cast(tf.argmax(network_answers[0], 1), tf.int32, name='prediction')
        else:
            prediction = network_answers[0]

        prediction_correct = tf.equal(prediction, labels, name='prediction_correct')
        num_correct = tf.reduce_sum(tf.cast(prediction_correct, tf.float32), name='num_correct')

        num_total = tf.cast(tf.shape(labels)[0], tf.float32, name='num_total')
    training_error = tf.truediv(num_correct, num_total, name=name)
    return [training_error, prediction], {}


def count_class(network_answers, class_value, name, compute_prediction=False):
    with tf.variable_scope(name+"_"):
        if compute_prediction:
            prediction = tf.cast(tf.argmax(network_answers[0], 1), tf.int32, name='prediction')
        else:
            prediction = network_answers[0]
        prediction_equal_class = tf.equal(prediction, np.int32(class_value))
#    if (class_value==1):
#        tf.identity(prediction_equal_class, 'prediction')
    num_in_class = tf.reduce_sum(tf.cast(prediction_equal_class, tf.float32), name=name)
    return [num_in_class], {}


def softmax_xentropy(network, params):
    """
    cross entropy softmax.
    this creates a new placeholder called y, which is an integer.
    @param network: 
    @param params: 
    @return: 
    """
    num_classes = params['num_classes']
    labels = tf.placeholder(tf.int32, [None], name='y')
    network_answers, _ = mlp.fully_connected_layer(network[0], num_classes, activation=None, name='decision')

    sm_network_answer = tf.nn.softmax(network_answers[0], name='sm_decision')

    [accuracy, prediction], _ = class_accuracy_max([sm_network_answer], labels)
#    tf.identity(prediction, name='prediction')

    # debug
    count_class([prediction], 0, 'num_zero')
    count_class([prediction], 1, 'num_one')
    tf.reduce_max(network_answers[0], name='max_answer')
    tf.reduce_min(network_answers[0], name='min_answer')

    network_answer = network_answers[0]
    labels_one_hot = tf.one_hot(labels, num_classes, on_value=1.0, name='onehot_label')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=network_answer),
                          name='loss')
    return [loss]

def sampled_softmax_xentropy(network, params):
    """
    cross entropy softmax.
    this creates a new placeholder called y, which is an integer.
    @param network:
    @param params:
    @return:
    """
    truncated_normal = 'truncated_normal'
    num_classes = params['num_classes']
    labels = tf.placeholder(tf.int32, [None], name='y')

    in_node = network[0][0]
    input_shape = in_node.get_shape().as_list()

    n_inputs = int(np.prod(input_shape[1:]))
    if len(input_shape) > 2:
        in_node = tf.reshape(in_node, [-1, n_inputs])
        # TODO: log when mismatch

    # w * x + b
    # this is the w part
    W_shape = [n_inputs, num_classes]
    W_init = initializations.get(truncated_normal)(W_shape, stddev=0.35)
    W_regul = None
    W_name = 'W_sm'
    W = variables.variable(W_name, shape=W_shape, regularizer=W_regul,
                           initializer=W_init, trainable=True,
                           restore=True)

    # this is the b part
    b_shape = [num_classes]
    bias_init = initializations.get(truncated_normal)(b_shape, stddev=0.35)
    b = variables.variable('b', shape=b_shape, initializer=bias_init,
                               trainable=True, restore=True)

    out_node = tf.matmul(in_node, W)
    network_answer = tf.nn.bias_add(out_node, b, name='decision')
    sm_network_answer = tf.nn.softmax(network_answer, name='sm_decision')

    #    network_answers, _ = mlp.fully_connected_layer(network[0], num_classes, activation=None, name='decision')
    labels = tf.reshape(labels, [-1,1])
#    print(labels.get_shape().as_list())
#    logits = tf.matmul(inputs, tf.transpose(weights))
#    logits = tf.nn.bias_add(logits, biases)
#    labels_one_hot = tf.one_hot(labels, n_classes)
#    loss = tf.nn.softmax_cross_entropy_with_logits(
#        labels=labels_one_hot,
#        logits=logits)

#    labels_one_hot = tf.one_hot(labels, num_classes, on_value=1.0, name='onehot_label')
#    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=network_answer),
#                          name='loss')
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
        weights=tf.transpose(W),
        biases=b,
        labels=labels,
        inputs=in_node,
        num_sampled=100,
        num_classes=num_classes,
        partition_strategy="div"),name='loss')
    return [loss]

def weighted_softmax_xentropy(network, params):
    """
    cross entropy softmax.
    this creates a new placeholder called y, which is an integer.
    @param network:
    @param params:
    @return:
    """
    num_classes = params['num_classes']
    labels = tf.placeholder(tf.int32, [None], name='y')
    network_answers, _ = mlp.fully_connected_layer(network[0], num_classes, activation=None, name='decision')

    sm_network_answer = tf.nn.softmax(network_answers[0], name='sm_decision')

    [accuracy, prediction], _ = class_accuracy_max([sm_network_answer], labels)
#    tf.identity(prediction, name='prediction')

    # debug
    count_class([prediction], 0, 'num_zero')
    count_class([prediction], 1, 'num_one')
    tf.reduce_max(network_answers[0], name='max_answer')
    tf.reduce_min(network_answers[0], name='min_answer')

    network_answer = network_answers[0]
    labels_one_hot = tf.one_hot(labels, num_classes, on_value=1.0, name='onehot_label')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=network_answer),
                          name='loss')
    return [loss]
#def hinge(network, params)

def l2_loss(network, params):
    labels = tf.placeholder(tf.float32, [None], name='y')
    return [tf.nn.l2_loss(tf.subtract(network[0], labels))]
