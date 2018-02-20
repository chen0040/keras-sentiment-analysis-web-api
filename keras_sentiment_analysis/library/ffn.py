from keras.layers import Dense, Dropout
from keras.models import model_from_json, Sequential
import numpy as np

from keras_sentiment_analysis.library.utility.glove_loader import GloveModel
from keras_sentiment_analysis.library.utility.tokenizer_utils import word_tokenize


class WordVecGloveFFN(object):


    def __init__(self):
        self.model = None
        self.glove_model = GloveModel()
        self.config = None
        self.word2idx = None
        self.idx2word = None
        self.max_len = None
        self.config = None
        self.vocab_size = None
        self.labels = None

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecGloveFFN.model_name + '_weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecGloveFFN.model_name + '_config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecGloveFFN.model_name + '_architecture.json'

    def load_model(self, model_dir_path):
        json = open(self.get_architecture_file_path(model_dir_path), 'r').read()
        self.model = model_from_json(json)
        self.model.load_weights(self.get_weight_file_path(model_dir_path))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        config_file_path = self.get_config_file_path(model_dir_path)

        self.config = np.load(config_file_path).item()

        self.idx2word = self.config['idx2word']
        self.word2idx = self.config['word2idx']
        self.max_len = self.config['max_len']
        self.vocab_size = self.config['vocab_size']
        self.labels = self.config['labels']

    def load_glove_model(self, data_dir_path, embedding_dim=None):
        self.glove_model.load(data_dir_path, embedding_dim=embedding_dim)

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', input_dim=self.glove_model.embedding_dim))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=2, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def predict(self, sentence):

        tokens = [w.lower() for w in word_tokenize(sentence)]

        X = np.zeros(shape=(1, self.glove_model.embedding_dim))
        E = np.zeros(shape=(self.glove_model.embedding_dim, self.max_len))
        for j in range(0, len(tokens)):
            word = tokens[j]
            try:
                E[:, j] = self.glove_model.encode_word(word)
            except KeyError:
                pass
        X[0, :] = np.sum(E, axis=1)
        output = self.model.predict(X)
        negative, positive = output[0]
        return [positive, negative]

    def test_run(self, sentence):
        print(self.predict(sentence))


def main():
    app = WordVecGloveFFN()
    app.test_run('i liked the Da Vinci Code a lot.')
    app.test_run('i hated the Da Vinci Code a lot.')

if __name__ == '__main__':
    main()
