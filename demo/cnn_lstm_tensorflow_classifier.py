from random import shuffle

import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences

from keras_sentiment_analysis.library.utility.simple_data_loader import load_text_label_pairs
from keras_sentiment_analysis.library.utility.tokenizer_utils import word_tokenize


def main():
    random_state = 42
    np.random.seed(random_state)

    model_dir_path = './models'
    data_file_path = './data/umich-sentiment-train.txt'
    text_label_pairs = load_text_label_pairs(data_file_path)

    shuffle(text_label_pairs)

    config_file_path = './models/wordvec_cnn_lstm_config.npy'
    config = np.load(config_file_path).item()

    idx2word = config['idx2word']
    word2idx = config['word2idx']
    max_len = config['max_len']
    vocab_size = config['vocab_size']
    labels = config['labels']

    with tf.gfile.FastGFile('./models/tf/wordvec_cnn_lstm.pb', 'rb') as f:
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
