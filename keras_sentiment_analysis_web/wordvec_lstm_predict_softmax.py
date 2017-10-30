from keras.models import model_from_json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import nltk


class WordVecLstmSoftmax(object):
    model = None
    word2idx = None
    idx2word = None
    context = None

    def __init__(self):
        json = open('../keras_sentiment_analysis/models/lstm_softmax_architecture.json', 'r').read()
        self.model = model_from_json(json)
        self.model.load_weights('../keras_sentiment_analysis/models/lstm_softmax_weights.h5')
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        self.word2idx = np.load('../keras_sentiment_analysis/models/umich_idx2word_lstm.npy').item()
        self.idx2word = np.load('../keras_sentiment_analysis/models/umich_word2idx_lstm.npy').item()
        self.context = np.load('../keras_sentiment_analysis/models/umich_context_lstm.npy').item()

    def predict(self, sentence):
        xs = []
        max_len = self.context['maxlen']
        tokens = [w.lower() for w in nltk.word_tokenize(sentence)]
        wid = [self.word2idx[token] if token in self.word2idx else 1 for token in tokens]
        xs.append(wid)
        x = pad_sequences(xs, max_len)
        output = self.model.predict(x)
        neg, pos = output[0]
        return [pos, neg]

    def test_run(self, sentence):
        print(self.predict(sentence))


if __name__ == '__main__':
    app = WordVecLstmSoftmax()
    app.test_run('i liked the Da Vinci Code a lot.')
    app.test_run('I hate friday')
