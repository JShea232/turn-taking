"""
Author: Jordan Shea
Written for Python 3.6.3.
"""

import argparse
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from main import train_test_split

def train_model(sentence_list, endings_list):
    """
    This function works to train a model utilizing multiple classifiers.
    For each classifier, the program will re-train to produce a new model,
    and see if this new model will result in a higher accuracy rating for
    a randomly generated test-set. As of right now, the models that are
    being utilized by this function include...

        1). LinearSVC model

    Once a model has been chosen, it will be saved to a pickle file that
    can then later be re-loaded to test on a sample data set.

    :param sentence_list: list of sample utterances
    :param endings_list: list of whether or not an utterance was interrupted
    :return: None
    """

    # Best models will initialize as None
    best_vectorizer = None
    best_accuracy = 0.0
    best_model = None

    for i in range(5):
        vectorizer = TfidfVectorizer()
        transformed_sentences = vectorizer.fit_transform(sentence_list)

        # Randomly generate a training and test set to get an accuracy
        training, testing, train_label, test_label = \
            train_test_split(transformed_sentences, endings_list, test_size=0.25)

        # Dictionary of classifiers that will 'compete' for the best
        # accuracy rating
        classifiers = {
            'LinearSVC-5': LinearSVC(C=5.0)
        }
        current_accuracy = 0.0
        current_model = None

        # Iterate through each possible classifier
        for name, clf in classifiers.items():
            clf.fit(training, train_label)
            predictions = clf.predict(testing)
            if current_accuracy < accuracy_score(test_label, predictions):
                current_accuracy = accuracy_score(test_label, predictions)
                current_model = clf

        # Update if a new best model has been selected
        if best_accuracy < current_accuracy:
            best_accuracy = current_accuracy
            best_model = current_model
            best_vectorizer = vectorizer
            print("Improved training accuracy of " + str(best_accuracy) + "%")

    print("Model chosen for fitted data...")
    print(best_model)

    # Store model and vectorizer into a pickle file
    with open('turn_taking_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    with open('turn_taking_vector.pkl', 'wb') as file:
        pickle.dump(best_vectorizer, file)

def test_model(sentences, endings):
    """
    This function works to evaluate a series of sentences to determine
    whether or not they were interrupted, or were able to complete
    naturally. If an utterance did NOT complete naturally, this may be
    indicative of certain features that can cause someone's turn to be
    interrupted more often.

    :param sentences: list of sample utterances
    :param endings: list of whether or not an utterance was interrupted
    :return: None
    """
    with open('turn_taking_model.pkl', 'rb') as file:
        clf = pickle.load(file)
    with open('turn_taking_vector.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    features = vectorizer.transform(sentences)
    predictions = clf.predict(features)
    print('Accuracy: %0.4f' % accuracy_score(endings, predictions))
    print('Precision: %0.4f' % precision_score(endings, predictions, average='macro'))
    print('Recall: %0.4f' % recall_score(endings, predictions, average='macro'))
    print('F1: %0.4f' % f1_score(endings, predictions, average='macro'))

def main():
    """Reads in transcript training data to build a suitable turn-taking model"""
    parser = argparse.ArgumentParser(
        description='Trains and gives information for a turn-taking model.')
    parser.add_argument('transcript_file', help='File containing the collated '
                                                'utterance information.')
    parser.add_argument('-t', action='store_true', help='Enable training.', dest='train')
    args = vars(parser.parse_args())
    transcript_file = args['transcript_file']
    do_train = args['train']
    if not os.path.isfile(transcript_file):
        raise RuntimeError('The given file does not exist!')

    # Read in training/test data
    data = read_data(transcript_file)

    sentences = [sentence for file_id, turn_type, speaker, turn_num, utt_num, sentence,
        good_start, good_end in data]
    good_ends = [good_end for file_id, turn_type, speaker, turn_num, utt_num, sentence,
        good_start, good_end in data]
    sentences, test_sentences, endings, test_endings = train_test_split(
        sentences, good_ends, test_size=0.1, random_state=1311)

    if do_train:
        train_model(sentences, endings)

    test_model(test_sentences, test_endings)

if __name__ == '__main__':
    from main import read_data
    main()
