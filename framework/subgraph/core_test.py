import unittest
import os
import tensorflow as tf
import numpy as np
from framework.subgraph.core import input_data, rename_nodes, fully_connected_layer


class TestCoreSubgraphs(unittest.TestCase):
    """
  
    """
  
    def test_input_data_rename_nodes(self):
        """
        Test the input_data and rename_nodes functions
        :return:
        """
        with tf.Session() as sess:
            x,_ = input_data([], 'x')
            y,_ = rename_nodes(x, ['y'])
            self.assertEqual(sess.run(y, {x[0]:1.0})[0], 1.0)
      
#  def fully_connected_layer(in_nodes, layer_size,
#                            activation='relu', bias=True,
#                            weights_init='truncated_normal', bias_init='truncated_normal',
#                            regularizer=None, weight_decay=0.001, trainable=True,
#                            restore=True, reuse=False,
#                            name="fc"):

    def test_fully_connected_layer(self):
        with tf.Session() as sess:
            x,_ = input_data([1], 'x')
            y,_ = fully_connected_layer(x, 10)
            self.assertEqual(sess.run(y, {x[0]:1.0})[0], 1.0)
      
    def test_fully_connected_network(self):
        None
    
    def test_combined(self):
        None
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    # just a dummy to load tensorflow
    unittest.main(verbosity=2)
