"""
Some convenience data functions
"""
import framework.utils.data.word_embedding as word_embedding

def load_embedding(filename='glove.6b.300d.pkl', dir='/tffs/data/'):
    embedding_file_list = [
         'glove.6b.300d.pkl'
        , 'w2v'
        , '1494172259337-Epoch-22.pkl'
        , '1493070821195-Epoch-64.pkl'
    ]
    if not (type(filename) is str):
        # assume it is integer
        filename = embedding_file_list[filename]
    embedding = word_embedding.WordEmbedding.from_file(dir+filename)
    return embedding