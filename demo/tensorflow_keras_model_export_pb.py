import os
import sys


def main():
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    from keras_sentiment_analysis.library.cnn_lstm import WordVecCnnLstm
    from keras_sentiment_analysis.library.cnn import WordVecCnn
    from keras_sentiment_analysis.library.cnn import WordVecMultiChannelCnn
    from keras_sentiment_analysis.library.lstm import WordVecLstmSoftmax
    from keras_sentiment_analysis.library.lstm import WordVecLstmSigmoid
    from keras_sentiment_analysis.library.lstm import WordVecBidirectionalLstmSoftmax

    classifier = WordVecCnnLstm()
    classifier.load_model(current_dir + '/models')
    classifier.export_tensorflow_model(output_fld=current_dir + '/models/tf')

    classifier = WordVecCnn()
    classifier.load_model(current_dir + '/models')
    classifier.export_tensorflow_model(output_fld=current_dir + '/models/tf')

    classifier = WordVecMultiChannelCnn()
    classifier.load_model(current_dir + '/models')
    classifier.export_tensorflow_model(output_fld=current_dir + '/models/tf')

    classifier = WordVecBidirectionalLstmSoftmax()
    classifier.load_model(current_dir + '/models')
    classifier.export_tensorflow_model(output_fld=current_dir + '/models/tf')

    classifier = WordVecLstmSoftmax()
    classifier.load_model(current_dir + '/models')
    classifier.export_tensorflow_model(output_fld=current_dir + '/models/tf')

    classifier = WordVecLstmSigmoid()
    classifier.load_model(current_dir + '/models')
    classifier.export_tensorflow_model(output_fld=current_dir + '/models/tf')


if __name__ == '__main__':
    main()

