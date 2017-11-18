from keras.models import model_from_json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import nltk
import os
import sys
import zipfile
import urllib.request

GLOVE_MODEL = "../keras_sentiment_analysis_train/very_large_data/glove.6B.100d.txt"
EMBED_SIZE = 100


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_glove():
    if not os.path.exists(GLOVE_MODEL):
        if not os.path.exists('../keras_sentiment_analysis_train/very_large_data'):
            os.makedirs('../keras_sentiment_analysis_train/very_large_data')

        glove_zip = '../keras_sentiment_analysis_train/very_large_data/glove.6B.zip'

        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall('../keras_sentiment_analysis_train/very_large_data')
        zip_ref.close()


def load_glove():
    _word2em = {}
    file = open(GLOVE_MODEL, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em


class WordVecGloveFFN(object):
    model = None
    word2em = None
    context = None

    def __init__(self):
        self.word2em = load_glove()
        json = open('../keras_sentiment_analysis_train/models/glove_ffn_architecture.json', 'r').read()
        self.model = model_from_json(json)
        self.model.load_weights('../keras_sentiment_analysis_train/models/glove_ffn_weights.h5')
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.context = np.load('../keras_sentiment_analysis_train/models/umich_context_glove.npy').item()

    def predict(self, sentence):
        max_len = self.context['maxlen']
        tokens = [w.lower() for w in nltk.word_tokenize(sentence)]

        X = np.zeros(shape=(1, EMBED_SIZE))
        E = np.zeros(shape=(EMBED_SIZE, max_len))
        for j in range(0, len(tokens)):
            word = tokens[j]
            try:
                E[:, j] = self.word2em[word]
            except KeyError:
                pass
        X[0, :] = np.sum(E, axis=1)
        output = self.model.predict(X)
        negative, positive = output[0]
        return [positive, negative]

    def test_run(self, sentence):
        print(self.predict(sentence))


if __name__ == '__main__':
    app = WordVecGloveFFN()
    app.test_run('i liked the Da Vinci Code a lot.')
    app.test_run('i hated the Da Vinci Code a lot.')
