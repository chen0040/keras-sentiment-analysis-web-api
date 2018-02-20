from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
from keras_sentiment_analysis.library.cnn import WordVecCnn, WordVecMultiChannelCnn
from keras_sentiment_analysis.library.cnn_lstm import WordVecCnnLstm
from keras_sentiment_analysis.library.lstm import WordVecLstmSigmoid
from keras_sentiment_analysis.library.lstm import WordVecLstmSoftmax
from keras_sentiment_analysis.library.lstm import WordVecBidirectionalLstmSoftmax
from keras_sentiment_analysis.library.ffn import WordVecGloveFFN

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

wordvec_cnn_classifier = WordVecCnn()
wordvec_multichannel_cnn_classifier = WordVecMultiChannelCnn()
wordvec_cnn_lstm_classifier = WordVecCnnLstm()
lstm_sigmoid_c = WordVecLstmSigmoid()
lstm_softmax_c = WordVecLstmSoftmax()
bidirectional_lstm_softmax_c = WordVecBidirectionalLstmSoftmax()
ffn_glove_c = WordVecGloveFFN()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'


@app.route('/wordvec_cnn', methods=['POST', 'GET'])
def wordvec_cnn():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            sentiments = wordvec_cnn_classifier.predict(sent)
            return render_template('wordvec_cnn_result.html', sentence=sent, sentiments=sentiments)
    return render_template('wordvec_cnn.html')


@app.route('/wordvec_cnn_lstm', methods=['POST', 'GET'])
def wordvec_cnn_lstm():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            sentiments = wordvec_cnn_lstm_classifier.predict(sent)
            return render_template('wordvec_cnn_lstm_result.html', sentence=sent, sentiments=sentiments)
    return render_template('wordvec_cnn_lstm.html')


@app.route('/lstm_sigmoid', methods=['POST', 'GET'])
def lstm_sigmoid():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            positive_sentiment, negative_sentiment = lstm_sigmoid_c.predict(sent)
            return render_template('lstm_sigmoid_result.html', sentence=sent,
                                   sentiments=[positive_sentiment, negative_sentiment])
    return render_template('lstm_sigmoid.html')


@app.route('/lstm_softmax', methods=['POST', 'GET'])
def lstm_softmax():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            sentiments = lstm_softmax_c.predict(sent)
            return render_template('lstm_softmax_result.html', sentence=sent,
                                   sentiments=sentiments)
    return render_template('lstm_softmax.html')


@app.route('/bidirectional_lstm_softmax', methods=['POST', 'GET'])
def bidirectional_lstm_softmax():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            sentiments = bidirectional_lstm_softmax_c.predict(sent)
            return render_template('bidirectional_lstm_softmax_result.html', sentence=sent,
                                   sentiments=sentiments)
    return render_template('bidirectional_lstm_softmax.html')


@app.route('/ffn_glove', methods=['POST', 'GET'])
def ffn_glove():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            sentiments = ffn_glove_c.predict(sent)
            return render_template('ffn_glove_result.html', sentence=sent,
                                   sentiments=sentiments)
    return render_template('ffn_glove.html')


@app.route('/measure_sentiments', methods=['POST', 'GET'])
def measure_sentiment():
    if request.method == 'POST':
        if not request.json or 'sentence' not in request.json or 'network' not in request.json:
            abort(400)
        sentence = request.json['sentence']
        network = request.json['network']
    else:
        sentence = request.args.get('sentence')
        network = request.args.get('network')

    sentiments = []
    if network == 'cnn':
        sentiments = wordvec_cnn_classifier.predict(sentence)
    elif network == 'cnn_lstm':
        sentiments = wordvec_cnn_lstm_classifier.predict(sentence)
    elif network == 'lstm_sigmoid':
        positive_sentiment = lstm_sigmoid_c.predict(sentence)[0]
        sentiments = [positive_sentiment, 1 - positive_sentiment]
    elif network == 'lstm_softmax':
        sentiments = lstm_softmax_c.predict(sentence)
    elif network == 'lstm_bidirectional_softmax':
        sentiments = bidirectional_lstm_softmax_c.predict(sentence)
    elif network == 'ffn_glove':
        sentiments = ffn_glove_c.predict(sentence)
    return jsonify({
        'sentence': sentence,
        'pos_sentiment': float(str(sentiments[0])),
        'neg_sentiment': float(str(sentiments[1])),
        'network': network
    })


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def main():
    model_dir_path = '../demo/models'

    wordvec_multichannel_cnn_classifier.load_model(model_dir_path)
    wordvec_cnn_lstm_classifier.load_model(model_dir_path)
    lstm_sigmoid_c.load_model(model_dir_path)
    wordvec_cnn_classifier.load_model(model_dir_path)
    lstm_softmax_c.load_model(model_dir_path)
    bidirectional_lstm_softmax_c.load_model(model_dir_path)

    ffn_glove_c.load_glove_model('../demo/very_large_data')
    ffn_glove_c.load_model(model_dir_path)

    wordvec_multichannel_cnn_classifier.test_run('i liked the Da Vinci Code a lot.')
    wordvec_cnn_lstm_classifier.test_run('i liked the Da Vinci Code a lot.')
    lstm_sigmoid_c.test_run('i liked the Da Vinci Code a lot.')
    wordvec_cnn_classifier.test_run('i liked the Da Vinci Code a lot.')
    lstm_softmax_c.test_run('i liked the Da Vinci Code a lot.')
    bidirectional_lstm_softmax_c.test_run('i like the Da Vinci Code a lot.')
    ffn_glove_c.test_run('i like the Da Vinci Code a lot.')
    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
