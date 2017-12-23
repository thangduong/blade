import unittest
import word_embedding as wb
import os
import tensorflow as tf
import numpy as np

class TestWordEmbedding(unittest.TestCase):
    """ Unit test class to test the WordEmbedding class
    """

    @classmethod
    def setUpClass(cls):
        cls._data_dir = os.environ.get('AGI_DATA_DIR')
        cls._glove_20151209 = wb.WordEmbedding.from_pkl_file(cls._data_dir + "/glove_20151209_vectors.txt.pkl", cls._data_dir + "/glove_20151209_vocab.txt")
        cls._body_glove_70k = wb.WordEmbedding.from_txt_file(cls._data_dir + "/70k_body_glove_vectors.txt", cls._data_dir + "/70k_body_glove_vocab.txt")

    def test_load(self):
        self.assertEqual(self._glove_20151209.word_count(), 1000001)
        self.assertEqual(self._glove_20151209.embedding_dimension_count(), 200)
        self.assertEqual(self._body_glove_70k.word_count(), 71971)
        self.assertEqual(self._body_glove_70k.embedding_dimension_count(), 50)

    def test_lookup(self):
        actual_vector = [0.209622, 0.255631, 1.018496, -0.076405, 0.621691, -0.423436, -0.585114, 0.267924, -0.466691, 1.357589, -0.074365, 0.052986, 0.022866, 1.35196, -0.086509, 0.247868, -0.220642, -0.221576, -0.637672, 0.44017, -1.424127, 0.825739, -0.015541, -0.907622, 0.101667, -0.100356, -0.253897, 0.4455, -0.069818, -0.629429, -1.563686, -1.218469, -0.27983, -0.848001, 0.971997, -0.590767, -0.158623, -0.309651, 1.026498, -0.008996, -0.302592, -0.340451, 0.022309, 0.196661, 0.45028, -0.280069, 0.878625, -0.554519, 0.562731, 0.175372]
        body_glove_70k_lookup = self._body_glove_70k.lookup_word('hi')

        self.assertEqual(len(actual_vector), len(body_glove_70k_lookup))
        for (actual_elt, lookup_elt) in zip(actual_vector, body_glove_70k_lookup):
            self.assertAlmostEqual(actual_elt, lookup_elt)

    def test_wordlist(self):
        body_glove_70k_index_lookup = self._body_glove_70k.wordlist_to_index_vector(['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy' ,'dog'])
        actual_indices = [0, 904, 1069, 3575, 9938, 91, 0, 8739, 895]
        self.assertEqual(len(actual_indices), len(body_glove_70k_index_lookup))
        for (actual_elt, lookup_elt) in zip(actual_indices, body_glove_70k_index_lookup):
            self.assertEqual(actual_elt, lookup_elt)

    def test_sentence(self):
        body_glove_70k_index_lookup = self._body_glove_70k.sentence_to_index_vector('the quick brown fox jumped over the lazy dog')
        actual_indices = [0, 904, 1069, 3575, 9938, 91, 0, 8739, 895]
        self.assertEqual(len(actual_indices), len(body_glove_70k_index_lookup))
        for (actual_elt, lookup_elt) in zip(actual_indices, body_glove_70k_index_lookup):
            self.assertEqual(actual_elt, lookup_elt)

    def test_tf_embedding_lookup(self):
        embedding_node = self._body_glove_70k.tf_embedding_lookup_from_sentence('the quick brown fox jumped over the lazy dog')
        sess = tf.Session()
        with sess.as_default():
            embedding_tensor = (embedding_node.eval())
            self.assertEqual(type(embedding_tensor),np.ndarray)

if __name__ == '__main__':
    unittest.main()
