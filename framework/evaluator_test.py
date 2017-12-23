import unittest
import numpy as np
from framework.evaluator import Evaluator
import framework.utils.data.text_indexer as ti

#class TestEvaluator(unittest.TestCase):
#    def test_evaluator_load(self):

"""
TODO: remove this after unit tests have been written.
graph_file = '/home/thduon/body_encoder/models/maltanet/maltanet.pb'
var_file = '/home/thduon/body_encoder/models/maltanet/maltanet_model.ckpt'
vocab_file = '/home/thduon/body_encoder/malta_data_vocab.pkl'
maltanet = Evaluator.load(graph_file, var_file)

def fit_to_width(input, width, filler = 0):
    if (len(input) >= width):
        output = input[0:width]
    else:
        zeros = np.zeros(width-len(input)).tolist()
        output = input + zeros
    return output

def create_input_string(indexer, input, width, filler=0):
    indexed = fit_to_width(indexer.index_text(input), width=width, filler=filler)
    return np.reshape(indexed,[1,width])

indexer = ti.TextIndexer.from_pkl_file(vocab_file)

query_width = 30
body_width = 250
query = create_input_string(indexer, 'is lion king a movie about a lion with a dead father ?', 30)
body1 = create_input_string(indexer, 'The lion king is a movie about a child lion who will become the king of the jungle . at the beginning of the movie , the father lion died .', body_width)
body2 = create_input_string(indexer,
                           'One confusing part about this is that the weights usually aren '' t stored inside the file format during training . Instead , they '' re held in separate checkpoint files , and there are Variable ops in the graph that load the latest values when they '' re initialized .', body_width)

#'encoded_query', 'encoded_body',
inputs = (maltanet.get_inputs())
[good_score] = maltanet.eval({'query':query, 'body':body1}, ['body_query_l2_norm'])
[bad_score] = maltanet.eval({'query':query, 'body':body2}, ['body_query_l2_norm'])
print('(good, bad) = (%s, %s)' % (good_score, bad_score))
"""