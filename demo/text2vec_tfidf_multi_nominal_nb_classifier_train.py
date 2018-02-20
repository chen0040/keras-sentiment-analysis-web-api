from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from keras_sentiment_analysis.library.utility.plot_utils import plot_confusion_matrix, most_informative_feature_for_binary_classification


def main():
    data_dir_path = './data'

    # Import `umich-sentiment-train.txt`
    df = pd.read_csv(data_dir_path + "/umich-sentiment-train.txt", sep='\t', header=None, usecols=[0, 1])

    print(df.head())

    Y = df[0].as_matrix()
    X = df[1].as_matrix()

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=53)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    clf = MultinomialNB()

    clf.fit(tfidf_train, y_train)
    pred = clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    plot_confusion_matrix(cm, classes=[0, 1])
    most_informative_feature_for_binary_classification(tfidf_vectorizer, clf, n=30)


if __name__ == '__main__':
    main()