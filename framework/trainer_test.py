import unittest
import tensorflow as tf
from framework.subgraph.distance import angle_distance, l2_norm
from framework.subgraph.core import input_data
import framework.utils.data.training_data as training_data
from trainer import Trainer
from framework.subgraph.mlp import fully_connected_network, fully_connected_layer

class TestTrainer(unittest.TestCase):
    """ Unit test class to test the WordEmbedding class
    """
    def test_dump_graph(self):
        def inference(params=None):
            x,_ = input_data([None, 5], name='x')
            y,_ = input_data([None, 5], name='y')
            x1, _ = fully_connected_network(x, [5,1], name='x1')
            y1, _ = fully_connected_network(y, [5,1], name='y1')
            d,_ = angle_distance(x1+y1, name='d')
            return d
        fake_data = training_data.generate_fake_1d_training_data(['x', 'y0'])
        trainer = Trainer(inference=inference, model_output_location='/tmp/dump_test',
                          name='dump_test', training_data=fake_data,
                          batch_size=16)
        trainer.dump_graph()

if __name__ == '__main__':
    unittest.main()
