from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier

from keras_sentiment_analysis.library.utility.glove_loader import GloveModel
from keras_sentiment_analysis.library.utility.plot_utils import plot_confusion_matrix


def main():
    data_dir_path = './data'

    # Import `umich-sentiment-train.txt`
    df = pd.read_csv(data_dir_path + "/umich-sentiment-train.txt", sep='\t', header=None, usecols=[0, 1])

    print(df.head())

    Y = df[0].as_matrix()
    X = df[1].as_matrix()

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    glove_model = GloveModel()
    glove_model.load('./very_large_data')
    X_train = glove_model.encode_docs(X_train)
    X_test = glove_model.encode_docs(X_test)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 2), random_state=42)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    plot_confusion_matrix(cm, classes=[0, 1])


if __name__ == '__main__':
    main()