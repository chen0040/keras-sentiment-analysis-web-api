from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

VERBOSE = 1


class MultiChannelCNNSentimentClassifier(object):
    EMBEDDING_SIZE = 100
    CNN_FILTER_SIZE = 32
    model_name = 'multi-channel-cnn'

    def __init__(self):
        self.loss_function = 'binary_crossentropy'
        self.model = None
        self.config = None
        self.max_input_tokens = None
        self.max_input_seq_length = None

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + os.path.sep + MultiChannelCNNSentimentClassifier.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + os.path.sep + MultiChannelCNNSentimentClassifier.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + os.path.sep + MultiChannelCNNSentimentClassifier.model_name + '-architecture.npy'

    def define_model(self, length, vocab_size, loss_function=None):
        if loss_function is not None:
            self.loss_function = loss_function

        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(vocab_size, MultiChannelCNNSentimentClassifier.EMBEDDING_SIZE)(inputs1)
        conv1 = Conv1D(filters=MultiChannelCNNSentimentClassifier.CNN_FILTER_SIZE, kernel_size=4, activation='relu')(embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)

        inputs2 = Input(shape=(length,))
        embedding2 = Embedding(vocab_size, MultiChannelCNNSentimentClassifier.EMBEDDING_SIZE)(inputs2)
        conv2 = Conv1D(filters=MultiChannelCNNSentimentClassifier.CNN_FILTER_SIZE, kernel_size=6, activation='relu')(embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)

        inputs3 = Input(shape=(length,))
        embedding3 = Embedding(vocab_size, MultiChannelCNNSentimentClassifier.EMBEDDING_SIZE)(inputs3)
        conv3 = Conv1D(filters=MultiChannelCNNSentimentClassifier.CNN_FILTER_SIZE, kernel_size=8, activation='relu')(embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)

        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(10, activation='relu')(merged)

        if loss_function == 'binary_crossentropy':
            outputs = Dense(1, activation='sigmoid')(dense1)
        else:
            outputs = Dense(1, activation='sigmoid')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        # compile
        model.compile(loss=self.loss_function, optimizer='adam', metrics=['accuracy'])
        # summarize
        print(model.summary())
        return model

    def fit(self, config, trainX, trainY, model_dir_path, loss_function=None, epochs=None, batch_size=None):
        if epochs is None:
            epochs = 10
        if batch_size is None:
            batch_size = 16
        self.config = config
        self.max_input_tokens = config['max_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']

        config_file_path = MultiChannelCNNSentimentClassifier.get_config_file_path(model_dir_path)
        np.save(config_file_path, config)

        model = self.define_model(self.max_input_seq_length, self.max_input_tokens, loss_function)

        weight_file_path = MultiChannelCNNSentimentClassifier.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)

        model.fit([trainX, trainX, trainX], trainY, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                  verbose=VERBOSE, callbacks=[checkpoint])
        # save the model
        model.save(weight_file_path)