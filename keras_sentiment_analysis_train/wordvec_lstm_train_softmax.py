import numpy as np
import collections
import nltk
from sklearn.cross_validation import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SpatialDropout1D, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

np.random.seed(42)

DATA_FILE = 'data/umich-sentiment-train.txt'

MAX_VOCAB_SIZE = 2000
BATCH_SIZE = 64
EPOCHS = 10
MAX_LENGTH = 40
EMBED_SIZE = 100

maxlen = 0
counter = collections.Counter()
with open(DATA_FILE, mode='rt', encoding='utf8') as file:
    for line in file:
        _, sentence = line.strip().split('\t')
        words = nltk.word_tokenize(sentence.lower())
        for word in words:
            counter[word] += 1

word2idx = collections.defaultdict(int)
for idx, word in enumerate(counter.most_common(MAX_VOCAB_SIZE)):
    word2idx[word[0]] = idx + 2
word2idx['UNK'] = 1
word2idx['PAD'] = 0
idx2word = {v: k for k, v in word2idx.items()}
vocab_size = len(word2idx)

np.save('models/umich_word2idx_lstm.npy', word2idx)
np.save('models/umich_idx2word_lstm.npy', word2idx)

context = {}
context['maxlen'] = MAX_LENGTH
np.save('models/umich_context_lstm.npy', context)

sx, sy = [], []
with open(DATA_FILE, mode='rt', encoding='utf8') as file:
    for line in file:
        label, sentence = line.strip().split('\t')
        words = nltk.word_tokenize(sentence.lower())
        wids = [word2idx[word] if word in word2idx else 1 for word in words]
        sx.append(wids)
        sy.append(int(label))
X = pad_sequences(sx, maxlen=MAX_LENGTH)
Y = np_utils.to_categorical(sy, 2)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, random_state=42, test_size=0.2)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=EMBED_SIZE, input_length=MAX_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=Xtrain, y=Ytrain, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=[Xtest, Ytest], verbose=1)

with open('models/lstm_softmax_architecture.json', 'w') as file:
    file.write(model.to_json())
model.save_weights('models/lstm_softmax_weights.h5')