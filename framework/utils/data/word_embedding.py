import logging
import pickle
import numpy as np
import tensorflow as tf
from framework.utils.data.text_indexer import TextIndexer
import math
import framework.utils.data.text_utils as utils
import heapq
import gensim

import warnings
warnings.filterwarnings('error')

def cos_distance(v1, v2):
    a = np.dot(v1, v2)
    b = np.dot(v1, v1)
    c = np.dot(v2, v2)
    result = a / math.sqrt(b*c)
    return result

def euclidean_distance(v1,v2):
    d = [x1-x2 for x1,x2 in zip(v1,v2)]
    d2 = np.dot(d,d)
    return math.sqrt(d2)

class WordEmbedding(object):
    """ Class to wrap up a word embedding
    """

    def __init__(self, embedding_matrix, vocab):
        """
        @param embedding_matrix: the embedding matrix
        @param vocab:  the vocabulary, which is a dict that maps from word to word index
        """
        # a matrix where each row is an embedding vector
        self._embedding_matrix = embedding_matrix
        # dictionary that maps from word to word index
        self._vocab = vocab
        self._indexer = TextIndexer(vocab)

    def debug_dump(self):
        print(self._embedding_matrix.shape)

    def get_indexer(self):
        """
        Get the indexer in this class
        @return:
        """
        return self._indexer

    def word_count(self):
        """
        @return:
         number of words in this embedding
        """
        return self._embedding_matrix.shape[0]

    def embedding_dimension_count(self):
        """
        @return:
         number of dimensions in the embedding space
        """
        return self._embedding_matrix.shape[1]

    def add_token(self, token, vector):
        """
        Add a token to the embedding by adding it to the vocab and matrix.
        Note that this is probably not optimal for numpy
        @param token: token to add
        @param vector: embedding vector for that token
        @return: None
        """
        if (not token in self._vocab):
            self._vocab[token] = self._embedding_matrix.shape[0]
            self._embedding_matrix = np.concatenate((self._embedding_matrix, vector))

    @classmethod
    def from_embedding_dict(cls, embedding_dict, vocab, unk_token=None, pad_token=None):
        """
        Create this class from the embedding dict and vocab
        @param embedding_dict:  dictionary that maps word to embedding vector
        @param vocab: dictionary that maps words to word index
        @return:
         WordEmbedding object
        """
        if not(unk_token==None) and not(unk_token in embedding_dict):
            vocab[unk_token] = len(vocab)
            embedding_dict[unk_token] = np.zeros(len(list(embedding_dict.values())[0]), dtype=np.float32)
        if not(pad_token==None) and not (pad_token in embedding_dict):
            vocab[pad_token] = len(vocab)
            embedding_dict[pad_token] = np.zeros(len(list(embedding_dict.values())[0]), dtype=np.float32)
        embedding_matrix = np.empty((len(embedding_dict), len(list(embedding_dict.values())[0])), dtype=np.float32)
        for word, vector in embedding_dict.items():
            if not word in vocab:
                print('word [%s] in embedding but not in vocab' % word)
                vocab[word] = len(vocab)
            embedding_matrix[vocab[word]] = vector
        return cls(embedding_matrix, vocab)

    @classmethod
    def load_vocab_file(cls, vocab_txt_file_name, embedding_dict):
        """
        Load the vocab from a txt file
        @param vocab_txt_file_name: the name of the text file containing the vocab.  This file is a space separated file
          that contains <vocab> <junk>
        @param embedding_dict: the embedding dictionary for sanity check.  words not in embedding_dict get discarded.
        @return:
        """
        index = 0
        vocab = {}
        with open(vocab_txt_file_name, 'rb') as vocab_txt_file:
            for line in vocab_txt_file:
                row = line.decode('utf8').strip().split(' ')
                word = row[0]
                count = row[1]
                if (word in embedding_dict):
                    vocab[word] = index
                    index = index+1
        return vocab

    @classmethod
    def from_pkl_file(cls, embedding_pkl_file_name, show_progress=True):
        """
        Load vocab and embedding data from pkl file
        @param vectors_pkl_file_name: name of pkl file
        @return: WordEmbedding object
        """
        if show_progress:
            print("loading " + embedding_pkl_file_name)
        with open(embedding_pkl_file_name, 'rb') as embedding_pkl_file:
            embedding_matrix,vocab = pickle.load(embedding_pkl_file)
            embedding = WordEmbedding(embedding_matrix, vocab)
            return embedding

    @classmethod
    def from_txt_file(cls, vectors_txt_file_name, unk_token='unk', pad_token='<pad>', show_progress=True, dimension=300):
        """
        Load embedding from txt file
        @param vectors_txt_file_name: txt file containing vectors
        @return: WordEmbedding object
        """
        vectors = {}
        vocab = {}
        word_index = 0
        with open(vectors_txt_file_name, 'rb') as vectors_txt_file:
            for i, line in enumerate(vectors_txt_file):
                if show_progress and (i%100000 == 0):
                    print("%s rows loaded" % i)
                line = line.rstrip().lstrip()
                row = line.decode('utf8').split(' ')

                # probably header
                if len(row) < dimension:
                    print("SKIPPING ROW %s" % i)
                    continue

                word = row[0]
                del row[0]
                vector = np.asarray(row).astype(np.float32)
                if not (len(vector) == dimension):
                    print('Error loading this line: ')
                    print(line)
                    row = line.decode('utf8').split(' ')
                    print(row)
                    exit(0)
                vectors[word] = vector
                vocab[word] = word_index
                word_index += 1
        embedding = WordEmbedding.from_embedding_dict(vectors, vocab, unk_token=unk_token, pad_token=pad_token)
        return embedding

    @classmethod
    def from_file(cls, filename):
        """
        Load embedding from file.  If file has .pkl, then use pickle.  If file has
        .txt, then use txt (load_txt_file)
        @param filename:  name of file
        @return:
        """
        if (filename.endswith('.pkl')):
            return cls.from_pkl_file(filename)
        elif (filename.endswith('.txt')):
            return cls.from_txt_file(filename)
        elif (filename.endswith('.bin')):
            return cls.from_bin_file(filename)
        else:
            logging.error('WordEmbedding.from_file: invalid filename "%s"' % filename)


    def get_vocab(self):
        """
        get the vocab dict
        @return:
        vocab dict which maps from word to index
        """
        return self._vocab

    def get_word_list(self):
        """
        get the list of words
        @return: 
        """
        return list(self._vocab.keys())

    def save_as_pkl_file(self, filename):
        """
        Save embedding matrix and vocab to a pickle file
        @param filename: name of pickle file
        @return:
        None
        """
        with open(filename, 'wb') as file:
            pickle.dump([self._embedding_matrix, self._vocab], file, protocol=4)

    def lookup_index(self, index):
        """
        Look up a word in the embedding matrix by index
        @param index: index of the word
        @return: embedding vector
        """
        return self._embedding_matrix[index]

    def wordlist_to_index_vector(self, wordlist, unk_word = 'unk', pad_token = '<pad>', min_len=0, max_len=0):
        """
        Convert a list of token/words to a list of corresponding indices based on indexer
        @param wordlist: list of words
        @param unk_word: what to use for unknown words
        @param pad_token: what to use for padding
        @param min_len: min len (pad if less)
        @param max_len: max len (truncate if more)
        @return:  list of indices
        """
        return self._indexer.index_wordlist(wordlist, unk_word, pad_token, min_len, max_len)

    def sentence_to_index_vector(self, sentence, unk_word = 'unk', pad_token = '<pad>', min_len=0, max_len=0):
        return self._indexer.index_text(sentence, unk_word, pad_token, min_len, max_len)

    def embedding_matrix(self):
        return self._embedding_matrix

    def _compute_context_vector(self, ref_word_v, ref_word, context_sentence, skip_self=True):
        word_list = utils.tokenize_text(context_sentence)
        context_vector = []
        midpt = int(len(ref_word_v) / 2)
        ref_word_dynamic = ref_word_v[midpt:]
        for w in word_list:
            if skip_self and (w == ref_word):
                continue
            v = self.lookup_word(w)
            if not (v is None):
                #                if (v.max() > 10000000):
                #                    print(w)
                #                    print(v)
                static_v = v[:midpt]
                context_vector.append(np.dot(static_v, ref_word_dynamic))
        return context_vector

    def lookup_word(self, word, context_sentence=None, context_skip_self=True,context_use_static=True):
        """
        Look up a word/token in the embedding        @param word:  word to look up
        @return: embedding vector
        """
        if word in self._vocab:
            if context_sentence is None:
                return self._embedding_matrix[self._vocab[word]]
            else:
                original = self.lookup_word(word)
                midpt = int(len(original) / 2)
                if context_use_static:
                    return np.append(original[:midpt], self._compute_context_vector(original, word, context_sentence,
                                                                                skip_self=context_skip_self))
                else:
                    return self._compute_context_vector(original, word, context_sentence,skip_self=context_skip_self)
        else:
            return None

    def context_lookup_word(self, word, context,context_use_static=True):
        """
        this is a faster version of lookup_word for context
        @param word:  word to look up
        @param context: a list of vector of context words
        @return: embedding vector
        """
        if word in self._vocab:
            original = self.lookup_word(word)
            midpt = int(len(original)/2)
            result = original[:midpt]
            context_vector = []
            for context_word_vector in context:
                context_vector.append(np.dot(result, context_word_vector[midpt:]))
            if context_use_static:
                return np.append(result, context_vector)
            else:
                return context_vector
        else:
            return None

    def vector_similarity(self, v1, v2, method=0):
        try:
            if method == 0:
                result = cos_distance(v1, v2)
            elif method == 1:
                result = euclidean_distance(v1, v2)
            elif method == 2:
                result = np.dot(v1, v2)
        except:
            result = -1
        return result

    def context_similarity(self, w1, w2, context, method=0, context_use_static=True, bits_to_use = 0):
        """
        This is a faster version of word_similary for context sensitive similarity.  The assumption is that
        the context words are already looked up.
        @param w1: 
        @param w2: 
        @param context: 
        @param method: 
        @return: 
        """
        v1 = self.context_lookup_word(w1,context,context_use_static=context_use_static)
        v2 = self.context_lookup_word(w2,context,context_use_static=context_use_static)
        midpt = int(len(v1) / 2)
        if bits_to_use == 1:
            v1 = v1[:midpt]
            v2 = v2[:midpt]
        if bits_to_use == 2:
            v1 = v1[midpt:]
            v2 = v2[midpt:]
        return self.vector_similarity(v1,v2, method)

    def word_similarity(self, w1, w2, method=0,
                        context_sentence1=None,
                        context_sentence2=None,
                        context_skip_self=True,
                        verbose=False,
                        context_use_static=True,
                        bits_to_use=1
                        ):
        """
        word similarity
        @param w1:  
        @param w2: 
        @param method:
           0 = cosine
           1 = euclidean
           2 = context dependent
        @return: 
        """
#        print('[%s,%s,%s]'%(w1,w2,context_sentence))
        result = 0
        v1 = self.lookup_word(w1,context_sentence=context_sentence1, context_skip_self=context_skip_self,context_use_static=context_use_static)
        if v1 is None:
            if verbose:
                print('WARNING: word %s is not in embedding' % w1.encode('utf-8'))
            return -1
#        if (v1.max()>10000000):
#            if verbose:
#                print(w1)
#                print(v1)
        v2 = self.lookup_word(w2,context_sentence=context_sentence2, context_skip_self=context_skip_self,context_use_static=context_use_static)
        if v2 is None:
            if verbose:
                print('WARNING: word %s is not in embedding' % w2.encode('utf-8'))
            return -1
#        if (v2.max()>10000000):
#            if verbose:
#                print(w2)
#                print(v2)
        midpt = int(len(v1) / 2)
        if bits_to_use == 1:
            v1 = v1[:midpt]
            v2 = v2[:midpt]
        if bits_to_use == 2:
            v1 = v1[midpt:]
            v2 = v2[midpt:]
        return self.vector_similarity(v1,v2,method)

    def get_most_similar(self, word, top_n = 10, method=0, verbose=True, context_sentence=None):
        """
        :param word: 
        :param top_n: 
        :param method: 
        :return: 
        """
        distances = []
        if verbose:
            num_words = len(self._vocab)
            dmark = num_words / 10
            mark = dmark
            print('looking through %s words' % num_words)

        if context_sentence is not None:
            context = []
            word_list = utils.tokenize_text(context_sentence)
            for context_word in word_list:
                if not(context_word == word):
                    context.append(self.lookup_word(context_word))

        for i, word2 in enumerate(self._vocab.keys()):
            if context_sentence is not None:
                distances.append({'word': word2,
                                  'distance': self.context_similarity(word, word2, context, method=method)})
            else:
                distances.append({'word': word2,
                          'distance': self.word_similarity(word,word2,method=method)})
            if verbose:
                if i > mark:
                    print("%s percent complete" % str(math.floor((float(i)/num_words)*1000)/10))
                    mark += dmark

        if not top_n==0:
            distances = heapq.nlargest(top_n, distances, key=lambda s: s['distance'])
        return distances

    def get_wordvec_dict(self):
        result = {}
        for word, idx in self._vocab.items():
            result[word] = self._embedding_matrix[idx]
        return result

    def tf_embedding_lookup(self, ids, partition_strategy='mod', name=None, validate_indices=True):
        return tf.nn.embedding_lookup(self._embedding_matrix, ids, partition_strategy, name, validate_indices)

    def tf_embedding_lookup_from_wordlist(self, wordlist, unk_word = 'unk', partition_strategy='mod', name=None, validate_indices=True):
        return tf.nn.embedding_lookup(self._embedding_matrix, self.wordlist_to_index_vector(wordlist, unk_word), partition_strategy, name, validate_indices)

    def tf_embedding_lookup_from_sentence(self, sentence, unk_word = 'unk', partition_strategy='mod', name=None, validate_indices=True):
        return tf.nn.embedding_lookup(self._embedding_matrix, self.sentence_to_index_vector(sentence, unk_word), partition_strategy, name, validate_indices)

    def tf_embedding_variable(self, name="word_embedding", trainable=False):
        return tf.get_variable(name, initializer=self._embedding_matrix, trainable=trainable, dtype=tf.float32)
