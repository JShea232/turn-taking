"""
Author: Jordan Shea
Written for Python 3.6.3.
"""

import argparse
import os

from main import read_data
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


vectorizer = TfidfVectorizer()


def train_model(sentence_list, endings_list):

    best_accuracy = 0.0
    best_model = None

    for i in range(5):
        transformed_sentences = vectorizer.fit_transform(sentence_list)
        training, testing, train_label, test_label = train_test_split(transformed_sentences, endings_list)
        classifiers = {
            'LinearSVC-5': LinearSVC(C=5.0)
        }
        current_accuracy = 0.0
        current_model = None

        for name in classifiers.keys():
            clf = classifiers[name]
            clf.fit(training, train_label)
            predictions = clf.predict(testing)
            if current_accuracy < accuracy_score(test_label, predictions):
                current_accuracy = accuracy_score(test_label, predictions)
                current_model = clf

        if best_accuracy < current_accuracy:
            best_accuracy = current_accuracy
            best_model = current_model
            print("Improved training accuracy of " + str(best_accuracy) + "%")

    print("Model chosen for fitted data...")
    print(best_model)

    return best_model


def test_model(clf, sentences, endings):

    features = vectorizer.transform(sentences)
    predictions = clf.predict(features)
    print('Turn-Taking Accuracy: %0.4f' % accuracy_score(endings, predictions))


def main():
    # Reads in transcript training data to build a suitable turn-taking model
    parser = argparse.ArgumentParser(
        description='Trains and gives information for a turn-taking model.')
    parser.add_argument('transcript_file', help='File containing the collated '
                                                'utterance information.')
    args = vars(parser.parse_args())
    transcript_file = args['transcript_file']
    if not os.path.isfile(transcript_file):
        raise RuntimeError('The given file does not exist!')
    data = read_data(transcript_file)
    sentences = []
    endings = []
    for entry in data[:100000]:
        sentences.append(entry[5])
        endings.append(entry[7])
    turn_taking_model = train_model(sentences, endings)

    test_sentences = []
    test_endings = []
    for entry in data[100000:]:
        test_sentences.append(entry[5])
        test_endings.append(entry[7])
    test_model(turn_taking_model, test_sentences, test_endings)


if __name__ == "__main__":
    main()
