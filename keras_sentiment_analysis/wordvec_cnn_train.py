import nltk

DATA_FILE = 'data/umich-sentiment-train.txt'
BATCH_SIZE = 64
NUM_EPOCHS = 20

file = open(DATA_FILE, mode='rt', encoding='utf8')
for line in file:
    _, sentence = line.strip().split('\t')
