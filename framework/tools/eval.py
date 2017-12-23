#!/usr/bin/python3
# evaluate a model
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from urllib.parse import unquote
from framework.evaluator import Evaluator
from framework.utils.data.text_indexer import TextIndexer
from framework.utils.data.word_embedding import WordEmbedding
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


tf.app.flags.DEFINE_string("model", "tunji_trec", "running mode: tf, tfdistributed")
tf.app.flags.DEFINE_string("output_format", "json", "output format.  valid options: json, plain")
tf.app.flags.DEFINE_string("embedding_filepath", "C:\\digitalcortex\\mrcmodels.thang_sat\\data\\glove.6b.300d.pkl", "embedding")
#tf.app.flags.DEFINE_string("inputs", "", "list of inputs")
#tf.app.flags.DEFINE_string("outputs", "", "list of outputs to get")
if os.name == 'nt':
  tf.app.flags.DEFINE_string("model_dir", "c:\\tmp", "comma separated list of parameter servers")
else:
  tf.app.flags.DEFINE_string("model_dir", "/tmp/", "comma separated list of parameter servers")

FLAGS = tf.app.flags.FLAGS
