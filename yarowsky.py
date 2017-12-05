"""
Author: Jordan Shea
Written for Python 3.6.3.
"""

import argparse
import math
import operator
import os

from main import read_data

CONSTANT = 0.2

# Dictionary used by the Yarowsky classifier
collocation_dict = {
    'BAD_END': {
        'first_word': {},
        'last_word': {},
        'sentence_length': {},
        'total': 0
    },
    'GOOD_END': {
        'first_word': {},
        'last_word': {},
        'sentence_length': {},
        'total': 0
    }
}


def calculate_best_collocation(decision_list, quantity):
    """
    This function works to print the best observed collocations and their
    probabilities to the user. This function takes in a decision list that is
    generated in the function create_decision_lists()

    :param decision_list: list of tuples in the format of..
                          (feature, word, probability)
    :param quantity: number of features to be printed out to the user
    :return: sorted version of the decision list
    """
    decision_list.sort(key=lambda tup: tup[2], reverse=True)
    print("BEST " + str(quantity) + " FEATURES FOR CLASSIFICATION")
    print("-------------------------------------------")
    for index in range(quantity):
        if index < len(decision_list):
            print(decision_list[index])
    print("-------------------------------------------")

    return decision_list


def compute_log(first_prob, second_prob):
    """
    This function works to compute Yarowsky's log probability given two
    collocation probabilities. If either or the probabilities is equal to zero,
    a small constant is added to both the numerator and the denominator in order
    to process the fraction.

    :param first_prob: first collocation probability >= 0
    :param second_prob: second collocation probability >= 0
    :return: Yarowsky's probability
    """
    if first_prob == 0 or second_prob == 0:
        first_prob += CONSTANT
        second_prob += CONSTANT
    return abs(math.log10(first_prob / second_prob))


def create_decision_lists(first, second):
    """
    This function works to generate a decision list of probabilities given
    a dictionary of collocation occurrences for two different senses of a word.

    :param first: first dictionary of collocation occurrences
    :param second: second dictionary of collocation occurrences
    :return: an unordered list of tuples in the format of...
             (feature, word, probability)
    """
    first_total = first['total']
    second_total = second['total']
    decision_list = []
    for entry in first:
        if entry != 'total':
            for word in first[entry]:
                first_prob = first[entry][word] / first_total
                if word not in second[entry]:
                    second_prob = 0
                else:
                    second_prob = second[entry][word] / second_total
                probability = compute_log(first_prob, second_prob)
                decision_list.append((entry, word, probability))
            # this is to check outstanding collocations
            for word in second[entry]:
                if word not in first[entry]:
                    second_prob = second[entry][word] / second_total
                    probability = compute_log(0, second_prob)
                    decision_list.append((entry, word, probability))

    return decision_list


def train_classifier(sentence: list):
    """
    This function works to generate a dictionary of collocation frequencies
    given whether or not the utterance was interrupted...
        - first_word   (the words appearing directly before the target word)
        - last_word   (the words appearing directly after the target word)
        - num_words (the words appearing with 5 words of the target word)

    :param sentence: tuple containing the sentence and relevant attributes
    :return: None
    """

    text = sentence[5].lower().split()
    good_end = sentence[7]

    if good_end:
        end = 'GOOD_END'
    else:
        end = 'BAD_END'

    collocation_dict[end]['total'] += 1

    if text:
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
    # Reads in transcript data and analyzes it for collocation data
    parser = argparse.ArgumentParser(
        description='Trains and gives information for a '
                    'collocation-based model.')
    parser.add_argument('transcript_file', help='File containing the collated '
                                                'utterance information.')
    args = vars(parser.parse_args())
    transcript_file = args['transcript_file']
    if not os.path.isfile(transcript_file):
        raise RuntimeError('The given file does not exist!')
    data = read_data(transcript_file)
    counter = 0
    for sen in data:
        counter += 1
        train_classifier(sen)

    print("NUM ITEMS: " + str(counter))
    print_common_features('GOOD_END', 'last_word', 10)
    print_common_features('BAD_END', 'last_word', 10)
    print(collocation_dict['GOOD_END']['total'])
    print(collocation_dict['BAD_END']['total'])
    decision_list = create_decision_lists(collocation_dict['GOOD_END'],
                                          collocation_dict['BAD_END'])
    calculate_best_collocation(decision_list, 10)


if __name__ == '__main__':
    main()

