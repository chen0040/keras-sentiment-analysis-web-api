# keras-sentiment-analysis-web-api

Web api built on flask for keras-based sentiment analysis using Word Embedding, RNN and CNN

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

## Training (Optional)

As the trained models are already included in the "keras_sentiment_analysis_train/models" folder in the project, the training is
not required. However, if you like to tune the parameters and retrain the models, you can use the 
following command to run the training:

```bash
cd keras_sentiment_analysis_train
python wordvec_bidirectional_lstm_train_softmax.py
```

The above commands will train bidrectional lstm model with softmax activation on the "keras_sentiment_analysis_train/data/umich-sentiment-train.txt" 
dataset and store the trained model in "keras_sentiment_analysis_train/models/bidirectional_lstm_softmax_**"

If you like to train other models, you can use the same command above on another train python scripts:

* wordvec_lstm_train_softmax.py: lstm model with softmax and categorical crossentropy objective
* wordvec_lstm_train_sigmoid.py: lstm model with sigmoid and binary crossentropy objective
* wordvec_cnn_train.py: cnn model with softmax and categorical crossentropy objective
* wordvec_glove_train.py: glove word embedding layer with feed forward network model and categorical crossentropy objective

## Running Web Api Server

Goto keras_sentiment_analysis_web directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained classifiers:

* 1-D CNN with Word Embedding 
* Feedforward network with Glove Word Embedding
* LSTM with binary or category cross-entropy loss function
* Bi-directional LSTM/GRU with categorical cross-entropy loss function

## Invoke Web Api

To query the sentiments using web api, after the flask server is started, run the following curl POST query
in your terminal:

```bash
curl -H 'Content-Type application/json' -X POST -d '{"network":"network_type_here", "sentence":"your_sentence_here"}' http://localhost:5000/measure_sentiments
```

(Note that same results can be obtained by running a curl GET query to http://localhost:5000/measure_sentiments?sentence=your_sentence_here&network=network_type_here)

For example, you can get the sentiments for the sentence "i like the Da Vinci Code a lot." by running the following command:

```bash
curl -H 'Content-Type: application/json' -X POST -d '{"network":"lstm_bidirectional_softmax", "sentence":"i like the Da Vinci Code a lot."}' http://localhost:5000/measure_sentiments
```

And the following will be the json response:

```json
{
    "neg_sentiment": 0.0000434154,
    "network": "lstm_bidirectional_softmax",
    "pos_sentiment": 0.999957,
    "sentence": "i like the Da Vinci Code a lot."
}
```

Here are some examples to query sentiments using some other neural network models:

```bash
curl -H 'Content-Type: application/json' -X POST -d '{"network":"lstm_softmax", "sentence":"i like the Da Vinci Code a lot."}' http://localhost:5000/measure_sentiments
curl -H 'Content-Type: application/json' -X POST -d '{"network":"lstm_sigmoid", "sentence":"i like the Da Vinci Code a lot."}' http://localhost:5000/measure_sentiments
curl -H 'Content-Type: application/json' -X POST -d '{"network":"cnn", "sentence":"i like the Da Vinci Code a lot."}' http://localhost:5000/measure_sentiments
curl -H 'Content-Type: application/json' -X POST -d '{"network":"ffn_glove", "sentence":"i like the Da Vinci Code a lot."}' http://localhost:5000/measure_sentiments
```











