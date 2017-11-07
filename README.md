# keras-sentiment-analysis-web-api

Web api built on flask for keras-based sentiment analysis using Word Embedding, RNN and CNN

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

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
```











