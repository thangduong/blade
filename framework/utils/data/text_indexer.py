import pickle
import logging
import framework.utils.data.text_utils as utils
import gzip

class TextIndexer(object):
    """
    Class to convert a string to an array of integers based on vocabulary

    """
    def __init__(self, vocab):
        """
        constructor.  takes a vocab which is a dict that maps between word and index
        @param vocab: dict that maps from word/token to index
        """
        self._vocab = vocab
        self._index_to_word = {}
        for x,y in vocab.items():
            self._index_to_word[y] = x


    def save_vocab_as_pkl(self, pkl_filename, compress=False):
        """
        Save the vocab to a pkl file
        @param pkl_filename:
        @return: None
        """
        if compress:
            with gzip.open(pkl_filename, 'wb') as pickle_out:
                pickle.dump(self._vocab, pickle_out)
        else:
            with open(pkl_filename, 'wb') as pickle_out:
                pickle.dump(self._vocab, pickle_out)

    @classmethod
    def from_pkl_file(cls, vocab_pkl_filename, compress=False):
        """
        Load vocab from pkl file
        @param vocab_pkl_filename: name of the pkl file that contains the vocab dict
        @return: the indexer
        """
        try:
            # try to load as if an embedding file
            if compress:
                with gzip.open(vocab_pkl_filename, 'rb') as vocab_pkl_file:
                    embedding_matrix, vocab = pickle.load(vocab_pkl_file)
            else:
                with gzip.open(vocab_pkl_filename, 'rb') as vocab_pkl_file:
                    embedding_matrix, vocab = pickle.load(vocab_pkl_file)
        except:
            # if that fails, then just load vocab
            if compress:
                with gzip.open(vocab_pkl_filename, 'rb') as vocab_pkl_file:
                    vocab = pickle.load(vocab_pkl_file)
            else:
                with open(vocab_pkl_filename, 'rb') as vocab_pkl_file:
                    vocab = pickle.load(vocab_pkl_file)
        return cls(vocab)

    @classmethod
    def from_txt_file(cls, txt_file_name, show_progress=True, min_freq=0, max_size=-1):
        """
        @param txt_file_name: name of the text file that contains a bunch of vocab words
        @return: the indexer
        """
        vocab = {}
        with open(txt_file_name, 'r') as txt_file:
            for i, line in enumerate(txt_file):
                if show_progress and (i%100000 == 0):
                    print("%s rows loaded" % i)
                cols = line.split(' ')
                word = cols[0]
                if (i>max_size and max_size>0) or int(cols[1])<min_freq:
                    break
                else:
                    vocab[word] = i
        return cls(vocab)

    @classmethod
    def from_file(cls, file_name, min_freq=0, max_size=-1):
        """
        Load either a txt or pkl file based on extension
        @param file_name: name of file
        @return:
        """
        if (file_name.endswith('.txt')):
            return TextIndexer.from_txt_file(file_name,min_freq=min_freq, max_size=max_size, show_progress=True)
        elif (file_name.endswith('.pkl')):
            return TextIndexer.from_pkl_file(file_name)
        elif (file_name.endswith('.pkz')):
            return TextIndexer.from_pkl_file(file_name)
        else:
            logging.error('Invalid file "%s".  Must be .pkl or .txt' % file_name)

    def index_wordlist(self, wordlist, unk_word = 'unk', pad_token = '<pad>', min_len=0, max_len=0):
        """
        Index a list of words based on the vocabulary.  Pad if list is < max_len and truncate if list is
        > min_len.

        :param wordlist: list of tokens to index
        :param unk_word: token to use when it is unknown
        :param pad_token: pad the list with this token if it is not long enough
        :param min_len: min len of output if > 0
        :param max_len: max len of output if > 0
        :return: a vector of indices of each token in the vocab
        """
        result = []
        unk_count = 0
        unk_list = []
        for word in wordlist:
            if (word in self._vocab):
#                print('IN VOCAB')
                result.append(self._vocab[word])
            else:
#                print('NOT IN VOCAB')
                result.append(self._vocab[unk_word])
                unk_list.append(word)
                unk_count += 1
#                try:
#                    print('UNK: %s' % word)
#                except:
#                    print("cannot print unk")
            if min_len>0 and len(result)>= min_len:
                break

        # here, expand to max_len if needed
        if max_len>0 and len(result) < max_len and pad_token and pad_token in self._vocab:
            result = result + [self._vocab[pad_token]] * (max_len - len(result))
        return len(wordlist), result, unk_count, unk_list

    def index_text(self, text, unk_word = 'unk', pad_token = '<pad>', min_len=0, max_len=0):
        """
        Index text by first converting to wordless using utils.tokenize_text and then calling index_wordlist.
        @param text: the text string
        @param unk_word: token to use when the word is not in the vocab
        @param pad_token: token to use to pad
        @param min_len: min length of the output vector
        @param max_len: max length of the output vector
        @return: a vector of indices of each token in the vocab
        """
        wordlist = utils.tokenize_text(text)
        return self.index_wordlist(wordlist, unk_word=unk_word, pad_token=pad_token, min_len=min_len, max_len=max_len)

    def add_token(self, token):
        if not token in self._vocab:
            self._index_to_word[len(self._vocab)] = token
            self._vocab[token] = len(self._vocab)

    def vocab_size(self):
        return len(self._vocab)

    def vocab_map(self):
        return self._vocab

    def index_to_word(self):
        return self._index_to_word
