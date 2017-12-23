import os

import framework.utils.data.word_embedding as wb
import text_indexer as ti

if False:
    print("Loading glove embedding...")
    data_dir = os.environ.get('AGI_DATA_DIR')
    if (data_dir is None):
        data_dir = 'agi/projects/body_encoder/data/'
    print('Using data in ' + os.path.abspath(data_dir))
    embedding = wb.WordEmbedding.from_txt_file(data_dir + "vectors.txt")

    vocab = embedding.get_vocab()
    indexer = ti.TextIndexer(vocab)
else:
    indexer = ti.TextIndexer.from_pkl_file('malta_data_vocab.pkl')

indexer.save_as_pkl('/tmp/test.pkl')

print(indexer.index_text('hello, world!'))
