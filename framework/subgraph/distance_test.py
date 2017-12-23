import unittest
import tensorflow as tf
from distance import angle_distance, l2_norm
from core import input_data

class TestDistance(unittest.TestCase):
    """ Unit test class to test the WordEmbedding class
    """
    def test_l2_norm(self):
        """
        Test the angle distance tensorflow
        @return:
        """
        with tf.Session() as sess:
            x, _ = input_data([None, 2,2], 'x')
            z, _ = l2_norm([x], name='n')
            r = sess.run(z[0], \
                     {   x[0]: \
                                 [ [[1, 2], [3, 4]], \
                                   [[1, 2], [3, 5]]] \
                         \
                      } \
                     )
            expected_r = [5.47722578, 6.24499798]
            for a,b in zip(r,expected_r):
                self.assertAlmostEqual(a,b)

    def test_angle_distance(self):
        """
        Test the angle distance tensorflow
        @return:
        """
        with tf.Session() as sess:
            x, _ = input_data([None,2,2], 'x')
            y, _ = input_data([None,2,2], 'y')
            z, _ = angle_distance([x,y], name='angle_distance')
            r = sess.run(z, \
                     {x[0]: [[[1,0],[1,0]]], \
                      y[0]: [[[0,1],[0,1]]] \
                      } \
                     )
            self.assertEqual(r[0], 0)

if __name__ == '__main__':
    unittest.main()
