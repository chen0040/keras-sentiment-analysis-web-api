from random import shuffle
import os
import sys
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def main():
    random_state = 42
    np.random.seed(random_state)

    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    data_file_path = current_dir + '/data/umich-sentiment-train.txt'

    from keras_sentiment_analysis.library.utility.simple_data_loader import load_text_label_pairs
    from keras_sentiment_analysis.library.utility.tokenizer_utils import word_tokenize

    text_label_pairs = load_text_label_pairs(data_file_path)

    shuffle(text_label_pairs)

    config_file_path = current_dir + '/models/tf/bidirectional_lstm_softmax.csv'
    first_line = True
    max_len = 0
    word2idx = dict()
    with open(config_file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if first_line:
                first_line = False
                max_len = int(line.strip())
            else:
                if line.startswith('label'):
                    pass
                else:
                    word, idx = line.strip().split('\t')
                    idx = int(idx)
                    word2idx[word] = idx

    with tf.gfile.FastGFile(current_dir + '/models/tf/bidirectional_lstm_softmax.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        [print(n.name) for n in sess.graph.as_graph_def().node]
        predict_op = sess.graph.get_tensor_by_name('output_node0:0')

        for i in range(20):
            sentence, label = text_label_pairs[i]

            xs = []
            tokens = [w.lower() for w in word_tokenize(sentence)]
            wid = [word2idx[token] if token in word2idx else len(word2idx) for token in tokens]
            xs.append(wid)
            x = pad_sequences(xs, max_len)

            predicted = sess.run(predict_op, feed_dict={"embedding_1_input:0": x,
                                                        'spatial_dropout1d_1/keras_learning_phase:0': 0})

            print(predicted)


if __name__ == '__main__':
    main()
