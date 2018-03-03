from keras_sentiment_analysis.library.cnn_lstm import WordVecCnnLstm
from keras_sentiment_analysis.library.cnn import WordVecCnn
from keras_sentiment_analysis.library.cnn import WordVecMultiChannelCnn
from keras_sentiment_analysis.library.lstm import WordVecLstmSoftmax
from keras_sentiment_analysis.library.lstm import WordVecLstmSigmoid
from keras_sentiment_analysis.library.lstm import WordVecBidirectionalLstmSoftmax


def main():
    classifier = WordVecCnnLstm()
    classifier.load_model('./models')
    classifier.export_tensorflow_model(output_fld='./models/tf')

    classifier = WordVecCnn()
    classifier.load_model('./models')
    classifier.export_tensorflow_model(output_fld='./models/tf')

    classifier = WordVecMultiChannelCnn()
    classifier.load_model('./models')
    classifier.export_tensorflow_model(output_fld='./models/tf')

    classifier = WordVecBidirectionalLstmSoftmax()
    classifier.load_model('./models')
    classifier.export_tensorflow_model(output_fld='./models/tf')

    classifier = WordVecLstmSoftmax()
    classifier.load_model('./models')
    classifier.export_tensorflow_model(output_fld='./models/tf')

    classifier = WordVecLstmSigmoid()
    classifier.load_model('./models')
    classifier.export_tensorflow_model(output_fld='./models/tf')


if __name__ == '__main__':
    main()

