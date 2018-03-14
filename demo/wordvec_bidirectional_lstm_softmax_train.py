import numpy as np
import os
import sys


def main():
    random_state = 42
    np.random.seed(random_state)

    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    output_dir_path = current_dir + '/models'
    data_file_path = current_dir + '/data/umich-sentiment-train.txt'

    from keras_sentiment_analysis.library.lstm import WordVecBidirectionalLstmSoftmax
    from keras_sentiment_analysis.library.utility.simple_data_loader import load_text_label_pairs
    from keras_sentiment_analysis.library.utility.text_fit import fit_text

    text_data_model = fit_text(data_file_path)
    text_label_pairs = load_text_label_pairs(data_file_path)

    classifier = WordVecBidirectionalLstmSoftmax()
    batch_size = 64
    epochs = 20
    history = classifier.fit(text_data_model=text_data_model,
                             model_dir_path=output_dir_path,
                             text_label_pairs=text_label_pairs,
                             batch_size=batch_size, epochs=epochs,
                             test_size=0.3,
                             random_state=random_state)


if __name__ == '__main__':
    main()
