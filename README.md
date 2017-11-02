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



