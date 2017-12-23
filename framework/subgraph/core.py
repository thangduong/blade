"""
Core subgraphs

"""
import framework.utils.graph.activations as activations
import tensorflow as tf

def activation(activation_function_name = 'linear'):
    """

    @param activation_function_name:
    @return:
    """
    out_nodes = []
    for in_node in in_nodes:
        out_nodes.append(activations.get(activation_function_name)(in_node))
    return out_nodes, {}

def rnn_dropout(in_nodes, keep_probs, name="dropout"):
    """
    Create a dropout TF node for every in_nodes with the corresponding keep prob.  This version
    works for RNN by using DropoutWrapper.
    The drop out is automatically gated by the "is_training" placeholder.  If "is_training" is false,
    then keep_prop is automatically set to 1.0.

    @param in_nodes:
    @param keep_probs:
    @param name:
    @return:
    A list of TF nodes
    """

    with tf.variable_scope(name):
        # if a single number input, then make it a list of the same number
        if not isinstance(keep_probs, list):
            keep_probs = [keep_probs] * len(in_nodes)

        is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
        out_nodes = []
        for in_node, keep_prob in zip(in_nodes, keep_probs):
            keep_prop_tensor = tf.cond(is_training, lambda: tf.constant(keep_prob), lambda: tf.constant(1.0))
            out_nodes.append(tf.contrib.rnn.DropoutWrapper(in_node, output_keep_prob=keep_prop_tensor))
        return out_nodes, {}

def dropout(in_nodes, keep_probs, name="dropout"):
    """
    Create a dropout TF node for every in_nodes with the corresponding keep prob.
    The drop out is automatically gated by the "is_training" placeholder.  If "is_training" is false,
    then keep_prop is automatically set to 1.0.

    @param in_nodes:
    @param keep_probs:
    @param name:
    @return:
    A list of TF nodes
    """

    with tf.variable_scope(name):
        # if a single number input, then make it a list of the same number
        if not isinstance(keep_probs, list):
            keep_probs = [keep_probs] * len(in_nodes)

        is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
        out_nodes = []
        for in_node, keep_prob in zip(in_nodes, keep_probs):
#            keep_prop_tensor = tf.cond(is_training, lambda: tf.constant(keep_prob), lambda: tf.constant(1.0))
#            out_nodes.append(tf.nn.dropout(in_node, keep_prop_tensor))
             out_nodes.append(tf.cond(is_training, lambda:tf.nn.dropout(in_node, keep_prob), lambda:in_node))
        return out_nodes, {}

def rename_nodes(in_nodes, out_names):
    """
    Rename a bunch of nodes by creating an identity node for each incoming node with the appropriate name

    @param in_nodes: nodes to rename with identity
    @param out_names: new names.  This array needs to be the same size as in_nodes
    @return:
        out_nodes, {}
    """
    return [tf.identity(node,name=name) for node, name in zip(in_nodes, out_names)], {}

def input_data(shape, name, dtype=tf.float32):
    """
    Create a tensorflow placeholder

    @param shape: list of int
    @param name: str
    @param dtype: tensorflow type
    @return:
    """
    return [tf.placeholder(shape=shape, dtype=dtype, name=name)], {}

#def embedding(in_nodes, in_vars = {}):
