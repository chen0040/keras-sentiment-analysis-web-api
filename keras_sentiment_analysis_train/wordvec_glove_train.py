import nltk
import collections
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.cross_validation import train_test_split

DATA_FILE = 'data/umich-sentiment-train.txt'
# the glove model file is not included in the git due to its file size, but can be
# downloaded from https://nlp.stanford.edu/projects/glove/
GLOVE_MODEL = "very_large_data/glove.6B.100d.txt"

BATCH_SIZE = 64
NUM_EPOCHS = 20
MAX_VOCAB_SIZE = 5000
EMBED_SIZE = 100

np.random.seed(42)

max_len = 0
file = open(DATA_FILE, mode='rt', encoding='utf8')
counter = collections.Counter()
for line in file:
    _, sentence = line.strip().split('\t')
    tokens = [w.lower() for w in nltk.word_tokenize(sentence)]
    for token in tokens:
        counter[token] += 1
    max_len = max(max_len, len(tokens))
file.close()

word2idx = collections.defaultdict(int)
for idx, word in enumerate(counter.most_common(MAX_VOCAB_SIZE)):
    word2idx[word[0]] = idx + 1
word2idx['_UNK_'] = 0
idx2word = {v: k for k, v in word2idx.items()}

sx = []
sy = []
file = open(DATA_FILE, mode='rt', encoding='utf8')
for line in file:
    label, sentence = line.strip().split('\t')
    tokens = [w.lower() for w in nltk.word_tokenize(sentence)]
    wids = []
    for token in tokens:
        if token in word2idx:
            wids.append(word2idx[token])
        else:
            wids.append(0)
    sx.append(wids)
    sy.append(label)
W = pad_sequences(sx, maxlen=max_len)
Y = np_utils.to_categorical(sy, 2)


word2em = {}
file = open(GLOVE_MODEL, mode='rb')
for line in file:
    words = line.strip().split()
    word = words[0]
    embeds = np.array(words[1:], dtype=np.float32)
    word2em[word] = embeds
file.close()

X = np.zeros(shape=(W.shape[0], EMBED_SIZE))
for i in range(W.shape[0]):
    words = [idx2word[idx] for idx in W[i].tolist()]
    E = np.zeros(shape=(EMBED_SIZE, max_len))
    for j in range(max_len):
        word = words[j]
        try:
            E[:, j] = word2em[word]
        except KeyError:
            pass
    X[i, :] = np.sum(E, axis=1)


model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=EMBED_SIZE))
model.add(Dropout(0.2))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_data=[Xtest, Ytest])

score = model.evaluate(x=Xtrain, y=Ytrain, verbose=1)

print('score: ', score[0])
print('accuracy: ', score[1])






