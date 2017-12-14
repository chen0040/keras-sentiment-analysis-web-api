import nltk
import collections
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import LSTM
from keras.layers import SpatialDropout1D, MaxPooling1D, Dense
from keras.utils import np_utils
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)

DATA_FILE = 'data/umich-sentiment-train.txt'
BATCH_SIZE = 64
NUM_EPOCHS = 20
MAX_VOCAB_SIZE = 5000
EMBEDDING_SIZE = 100
LSTM_OUTPUT_SIZE = 70
WEIGHT_FILE_PATH = 'models/wordvec_cnn_lstm_weights.h5'

counter = collections.Counter()
file = open(DATA_FILE, mode='rt', encoding='utf8')
max_len = 0
for line in file:
    _, sentence = line.strip().split('\t')
    tokens = [x.lower() for x in nltk.word_tokenize(sentence)]
    for token in tokens:
        counter[token] += 1
    max_len = max(max_len, len(tokens))
file.close()

word2idx = collections.defaultdict(int)
for idx, word in enumerate(counter.most_common(MAX_VOCAB_SIZE)):
    word2idx[word[0]] = idx
idx2word = {v: k for k, v in word2idx.items()}
vocab_size = len(word2idx) + 1

context = {'maxlen': max_len }

np.save('models/umich_word2idx.npy', word2idx)
np.save('models/umich_idx2word.npy', idx2word)
np.save('models/umich_context.npy', context)

xs = []
ys = []
file = open(DATA_FILE, mode='rt', encoding='utf8')
for line in file:
    label, sentence = line.strip().split('\t')
    tokens = [x.lower() for x in nltk.word_tokenize(sentence)]
    wid = [word2idx[w] for w in tokens]
    xs.append(wid)
    ys.append(label)

X = pad_sequences(xs, maxlen=max_len)
Y = np_utils.to_categorical(ys, 2)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=42)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, input_length=max_len, output_dim=EMBEDDING_SIZE))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(LSTM_OUTPUT_SIZE))
model.add(Dense(units=2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
json = model.to_json()

open('models/wordvec_cnn_lstm_architecture.json', 'w').write(json)

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)

model.fit(x=Xtrain, y=Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=[Xtest, Ytest], verbose=1,
          callbacks=[checkpoint])

score = model.evaluate(x=Xtest, y=Ytest, batch_size=BATCH_SIZE, verbose=1)
print('score: ', score[0])
print('accuracy: ', score[1])


model.save_weights(WEIGHT_FILE_PATH)