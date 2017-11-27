"""
Author: Jordan Shea
Written for Python 3.6.3.
"""

import argparse
import operator
import os

from main import read_data

collocation_dict = {
    'BAD_START': {
        'first_word': {},
        'last_word': {},
        'sentence_length': {}
    },
    'BAD_END': {
        'first_word': {},
        'last_word': {},
        'sentence_length': {}
    },
    'GOOD_START': {
        'first_word': {},
        'last_word': {},
        'sentence_length': {}
    },
    'GOOD_END': {
        'first_word': {},
        'last_word': {},
        'sentence_length': {}
    }
}

def train_classifier(sentence: list):
    """Stores sentence information in the classifier.

    Args:
        sentence: An list of data for the utterance to process.
    """
    text = sentence[5].split()
    good_start = sentence[6]
    good_end = sentence[7]

    if good_start:
        start = 'GOOD_START'
    else:
        start = 'BAD_START'

    if good_end:
        end = 'GOOD_END'
    else:
        end = 'BAD_END'

    if text:
        # Build dictionary for good/bad stars
        if str(len(text)) in collocation_dict[start]['sentence_length']:
            collocation_dict[start]['sentence_length'][str(len(text))] += 1
        else:
            collocation_dict[start]['sentence_length'][str(len(text))] = 1

        if text[0] in collocation_dict[start]['first_word']:
            collocation_dict[start]['first_word'][text[0]] += 1
        else:
            collocation_dict[start]['first_word'][text[0]] = 1

        if text[-1] in collocation_dict[start]['last_word']:
            collocation_dict[start]['last_word'][text[-1]] += 1
        else:
            collocation_dict[start]['last_word'][text[-1]] = 1

        # Build dictionary for good/bad endings
        if str(len(text)) in collocation_dict[end]['sentence_length']:
            collocation_dict[end]['sentence_length'][str(len(text))] += 1
        else:
            collocation_dict[end]['sentence_length'][str(len(text))] = 1

        if text[0] in collocation_dict[end]['first_word']:
            collocation_dict[end]['first_word'][text[0]] += 1
        else:
            collocation_dict[end]['first_word'][text[0]] = 1

        if text[-1] in collocation_dict[end]['last_word']:
            collocation_dict[end]['last_word'][text[-1]] += 1
        else:
            collocation_dict[end]['last_word'][text[-1]] = 1

def print_common_features(sentence_type: str, feature: str, num_items: int):
    """Prints instances of a feature that occur often.

    Args:
        sentence_type: The type of sentence boundary to analyze.
        feature: The feature being analyzed.
        num_items: The number of most common instances of the feature to display.
    """
    my_dict = collocation_dict[sentence_type][feature]
    sorted_dict = sorted(my_dict.items(), key=operator.itemgetter(1), reverse=True)

    print('FEATURE: ' + feature + ' for ' + sentence_type)
    for index in range(num_items):
        entry = sorted_dict[index]
        print('KEY: ' + str(entry[0]) + ' -> VALUE: ' + str(entry[1]))
    print('------------------------------------')

def main():
    """Reads in transcript data and analyzes it for collocation data"""
    parser = argparse.ArgumentParser(
        description='Trains and gives information for a collocation-based model.')
    parser.add_argument('transcript_file', help='File containing the collated utterance information.')
    args = vars(parser.parse_args())
    transcript_file = args['transcript_file']
    if not os.path.isfile(transcript_file):
        raise RuntimeError('The given file does not exist!')
    data = read_data(transcript_file)
    for sen in data:
        train_classifier(sen)
    print_common_features('GOOD_END', 'first_word', 10)
    print_common_features('GOOD_END', 'last_word', 10)
    print_common_features('GOOD_END', 'sentence_length', 10)

    print_common_features('BAD_END', 'first_word', 10)
    print_common_features('BAD_END', 'last_word', 10)
    print_common_features('BAD_END', 'sentence_length', 10)

    print_common_features('GOOD_START', 'first_word', 10)
    print_common_features('GOOD_START', 'last_word', 10)
    print_common_features('GOOD_START', 'sentence_length', 10)

    #print_common_features('BAD_START', 'first_word', 10)
    #print_common_features('BAD_START', 'last_word', 10)
    #print_common_features('BAD_START', 'sentence_length', 10)

if __name__ == '__main__':
    main()
