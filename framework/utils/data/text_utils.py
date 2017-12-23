#import enchant
import nltk
from nltk.corpus import stopwords
import pickle
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add('since')

#english = enchant.Dict('en_US')
TOKEN_DELIMITERS = ['/','~','*','-','|','”','`','…','™','’','—',':','“','\\','=','°','+',',','"', '\'', '.','€'
, b'\xe2\x80\x80'.decode('utf-8')
, b'\xe2\x80\x81'.decode('utf-8')
, b'\xe2\x80\x82'.decode('utf-8')
, b'\xe2\x80\x83'.decode('utf-8')
, b'\xe2\x80\x84'.decode('utf-8')
, b'\xe2\x80\x85'.decode('utf-8')
, b'\xe2\x80\x86'.decode('utf-8')
, b'\xe2\x80\x87'.decode('utf-8')
, b'\xe2\x80\x88'.decode('utf-8')
, b'\xe2\x80\x89'.decode('utf-8')
, b'\xe2\x80\x8a'.decode('utf-8')
, b'\xe2\x80\x8b'.decode('utf-8')
, b'\xe2\x80\x8c'.decode('utf-8')
, b'\xe2\x80\x8d'.decode('utf-8')
, b'\xe2\x80\x8e'.decode('utf-8')
, b'\xe2\x80\x8f'.decode('utf-8')
, b'\xe2\x80\x90'.decode('utf-8')
, b'\xe2\x80\x91'.decode('utf-8')
, b'\xe2\x80\x92'.decode('utf-8')
, b'\xe2\x80\x93'.decode('utf-8')
, b'\xe2\x80\x94'.decode('utf-8')
, b'\xe2\x80\x95'.decode('utf-8')
, b'\xe2\x80\x96'.decode('utf-8')
, b'\xe2\x80\x97'.decode('utf-8')
, b'\xe2\x80\x98'.decode('utf-8')
, b'\xe2\x80\x99'.decode('utf-8')
, b'\xe2\x80\x9a'.decode('utf-8')
, b'\xe2\x80\x9b'.decode('utf-8')
, b'\xe2\x80\x9c'.decode('utf-8')
, b'\xe2\x80\x9d'.decode('utf-8')
, b'\xe2\x80\x9e'.decode('utf-8')
, b'\xe2\x80\x9f'.decode('utf-8')
, b'\xe2\x80\xa0'.decode('utf-8')
, b'\xe2\x80\xa1'.decode('utf-8')
, b'\xe2\x80\xa2'.decode('utf-8')
, b'\xe2\x80\xa3'.decode('utf-8')
, b'\xe2\x80\xa4'.decode('utf-8')
, b'\xe2\x80\xa5'.decode('utf-8')
, b'\xe2\x80\xa6'.decode('utf-8')
, b'\xe2\x80\xa7'.decode('utf-8')
, b'\xe2\x80\xa8'.decode('utf-8')
, b'\xe2\x80\xa9'.decode('utf-8')
, b'\xe2\x80\xaa'.decode('utf-8')
, b'\xe2\x80\xab'.decode('utf-8')
, b'\xe2\x80\xac'.decode('utf-8')
, b'\xe2\x80\xad'.decode('utf-8')
, b'\xe2\x80\xae'.decode('utf-8')
, b'\xe2\x80\xaf'.decode('utf-8')
, b'\xe2\x80\xb0'.decode('utf-8')
, b'\xe2\x80\xb1'.decode('utf-8')
, b'\xe2\x80\xb2'.decode('utf-8')
, b'\xe2\x80\xb3'.decode('utf-8')
, b'\xe2\x80\xb4'.decode('utf-8')
, b'\xe2\x80\xb5'.decode('utf-8')
, b'\xe2\x80\xb6'.decode('utf-8')
, b'\xe2\x80\xb7'.decode('utf-8')
, b'\xe2\x80\xb8'.decode('utf-8')
, b'\xe2\x80\xb9'.decode('utf-8')
, b'\xe2\x80\xba'.decode('utf-8')
, b'\xe2\x80\xbb'.decode('utf-8')
, b'\xe2\x80\xbc'.decode('utf-8')
, b'\xe2\x80\xbd'.decode('utf-8')
, b'\xe2\x80\xbe'.decode('utf-8')
, b'\xe2\x80\xbf'.decode('utf-8')
]

nltk_english_stopwords = set(stopwords.words('english'))


def _split_str_list(str_list, delim_list):
    """
    Split every string in a list of strings and merge all the pieces into one large list.
    For example:
    str_list: ['a@b', 'c,d', 'e!f']
    delim_list: ['@',',','!']
    Generates output:
    ['a','@','b','c',',','d','e','!','f']

    @param str_list: list of strings
    @param delim_list: list of delimiters
    @return: list of tokens from the strings
    """
    result_list = str_list
    for delim in delim_list:
        str_list = result_list
        result_list = []
        for str_in in str_list:
            str_in_split = str_in.split(delim)
            first = True
            for split_str in str_in_split:
                if first:
                    first = False
                    if (split_str != ''):
                        result_list.append(split_str)
                else:
                    result_list.append(delim)
                    if (split_str != ''):
                        result_list.append(split_str)
    return result_list

def tokenize_text(text, delimiters=TOKEN_DELIMITERS, discard_stopwords=False):
    """
    Tokenize a string based on nltk and delimiters
    @param text: text string to document
    @param delimiters:
    @return:
    """
    if delimiters==None:
        delimiters = TOKEN_DELIMITERS
    result = _split_str_list(nltk.word_tokenize(text), delimiters)
    if discard_stopwords:
        result = [x for x in result if x not in nltk_english_stopwords and x not in delimiters]
    return result

def normalize_text(text, return_string_separator=' ', language=None, min_token_len=0, exception_tokens=None):
    """
    Convert a text into lower case, split into tokens, and re-construct with ' ' in between
    @param text: the input text string to normalize
    @param return_string_separator: how to concatenate tokens in the return string.  If this is None, then return a list of tokens instead.
    @param language: what language to use.  If None, then use english.  This is for NLTK.
    @param min_word_len: minimum token length.  Tokens shorter than this will get discarded.
    @param exception_tokens: tokens that don't get discarded regardless of what conditions if failed to meet.
    @return:
     the normalized text string or a list of normalized tokens
    """
    tokens = tokenize_text(text)
    result_tokens = tokens
    if language or min_token_len>0 or exception_tokens:
        result_tokens = []
        for token in tokens:
            if ((not language) or language.check(token)) and (len(token)>min_token_len) or (exception_tokens and (token in exception_tokens)):
                result_tokens.append(token)
    if return_string_separator:
        return return_string_separator.join(result_tokens)
    else:
        return result_tokens

def count_tokens(token_freq_map, sentence, delimiters=None, add_new_tokens=True,skip_stopwords=False):
    token_list = tokenize_text(sentence, delimiters=delimiters)
    for token in token_list:
        if skip_stopwords and (token in stop_words or len(token)==1):
            continue
        if token in token_freq_map:
            token_freq_map[token]+=1
        elif add_new_tokens:
            token_freq_map[token]=1
    return len(token_list)

def count_tokens_in_file(filename, token_freq_map={}, delimiters=None, lower=True, add_new_tokens=True, num_lines=-1, checkpoint_file="", checkpoint_period=-1, skip_stopwords=False):
    total_tokens = 0
    with open(filename, "r", encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            line = line.rstrip().lstrip()
            if lower:
                line = line.lower()
            total_tokens += count_tokens(token_freq_map, line, delimiters, add_new_tokens=add_new_tokens, skip_stopwords=skip_stopwords)
            if line_no%1000 == 0:
                print('at line %s' % line_no)
            if num_lines > 0 and line_no > num_lines:
                break
            if len(checkpoint_file)>0 and checkpoint_period>0 and (line_no%checkpoint_period)==0:
                print('writing checkpoint to %s'%checkpoint_file)
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump([token_freq_map, total_tokens],f)
    return total_tokens, token_freq_map

def count_bigram_tokens(token_freq_map, sentence, delimiters=None, add_new_tokens=True):
    token_list = tokenize_text(sentence, delimiters=delimiters)
    for w1, w2 in zip(token_list[:-1], token_list[1:]):
        token = ' '.join([w1, w2])
        if token in token_freq_map:
            token_freq_map[token] += 1
        elif add_new_tokens:
            token_freq_map[token] = 1
    return len(token_list)

def count_bigrams_tokens_in_file(filename, token_freq_map={}, delimiters=None, lower=True, add_new_tokens=True):
    total_tokens = 0
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip().lstrip()
            if lower:
                line = line.lower()
            total_tokens += count_bigram_tokens(token_freq_map, line, delimiters, add_new_tokens=add_new_tokens)
    return total_tokens, token_freq_map

def dump_token_map(token_map):
    sorted_map = sorted(token_map.items(), key = lambda value: value[1], reverse=True)
    for (k, v) in sorted_map:
        print("%s %s" % (k, v))
    #    print(sorted_map)
