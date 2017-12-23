"""
Subgraphs that don't fit into any group
"""

import tensorflow as tf
from framework.subgraph import subgraph_helper as sghelper
import numpy as np

def polynomial(in_nodes, order, in_vars=None, name='polynomial'):
    """
    A subgraph to build a network that computes a polynomial function of the input
  
    Only in_nodes[0] is used.  This subgraph computes:
        <name>/output = a0 + a1 * in_nodes[0] + a1 * in_nodes[0]^2 + ...
    where a_j are scalar variables.
  
    :param in_nodes:
  
    :param order:
    :param in_vars:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        out_vars = {}
        a0 = sghelper.get_variable('a0', [], out_vars, in_vars=in_vars)
        terms = [tf.scalar_mul(a0, tf.ones_like(in_nodes[0]))]
        for i in range(1,order+1):
            terms.append(tf.scalar_mul(sghelper.get_variable('a%d'%i, [], out_vars, in_vars=in_vars), tf.pow(in_nodes[0], tf.constant(float(i)))))
        return [tf.add_n(terms, name='output')], out_vars

def concat(in_nodes):
    """
    Force concatenate all in_nodes together even if their dimensions don't match.
    Intput is a list of vectors of size [MB, N[j]].
    Output is a single tensor of size [MB, K]. 
    @param in_nodes: list of vectors
    @return: 
    """
    reshaped_nodes = []
    for node in in_nodes:
        reshaped_nodes.append(tf.reshape(node, [-1, int(np.prod(node.get_shape().as_list()[1:]))]))
    return [tf.concat(reshaped_nodes, 1)], {}