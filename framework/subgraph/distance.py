import tensorflow as tf
from framework.subgraph.core import rename_nodes

def l2_norm(innodes, name='l2norm', sqrt=True):
    """
    Return the L2 norms of all input nodes.
    @param innodes: list of input nodes
    @param name:
        all the sub pieces will be in [name]_.
        the outputs are [name]_j if there are multiple outputs.
        otherwise, the output is [name].
    @return:
        outnodes, None
    """
    outputs = []
    with tf.variable_scope('%s'%name):
        for innode in innodes:
            square = tf.square(innode)
            sum = tf.squeeze(tf.reduce_sum(square, list(range(1,len(square.get_shape())))))
            if sqrt:
                output = tf.sqrt(sum)
            else:
                output = sum
            outputs.append(output)

    # rename
    if len(innodes)>0:
        outputs, _ = rename_nodes(outputs, ['%s_%s'%(name,str(i)) for i in range(0,len(innodes))])
    else:
        outputs, _ = rename_nodes(outputs, [name])
    return outputs, None


def angle_distance(innodes, name='angle_distance'):
    """
    Returns the angle distance between two nodes.  Angle distance is <x,y>/(||x||||y||)
    @param innodes: array of 2 tensorflow tensors with equal dimensions
    @param name: name of the subgraph.  everything in this subgraph would be _[name], and the output node would be
    [name]
    @return:
     <tensor>, _
    """

    with tf.variable_scope('%s'%name):
        prod = tf.mul(innodes[0], innodes[1])
        numerator = tf.squeeze(tf.reduce_sum(prod, list(range(1,len(prod.get_shape())))), name='numerator')
        n,_ = l2_norm(innodes, 'l2_norm')
        denominator = tf.mul(n[0],n[1], name='denominator')
        # TODO: check for divide by 0
        output = tf.div(numerator, denominator, name='output')
    outputs = [output]
#    outputs, _ = rename_nodes([output], [name])
    return outputs, None
