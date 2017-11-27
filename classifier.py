"""
Author: Jordan Shea
Written for Python 3.6.3.
"""

import argparse
import os
import main


collocation_dict = {
    "BAD_START": {
        "first_word": {

        },
        "last_word": {

        },
        "sentence_length": {

        }
    },
    "BAD_END": {
        "first_word": {

        },
        "last_word": {

        },
        "sentence_length": {

        }
    },
    "GOOD_START": {
        "first_word": {

        },
        "last_word": {

        },
        "sentence_length": {

        }
    },
    "GOOD_END": {
        "first_word": {

        },
        "last_word": {

        },
        "sentence_length": {

        }
    }
}


def train_classifier(sentence: list):
    text = sentence[5].split()
    good_start = sentence[6]
    good_end = sentence[7]

    if good_start == "True":
        start = 'GOOD_START'
    else:
        start = 'BAD_START'

    if good_end == "True":
        end = 'GOOD_END'
    else:
        end = 'BAD_END'

    if len(text) > 0:

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


def print_common_features(sentence_type: str, feature: str, min_freq: int):
    print("FEATURE: " + feature + " for " + sentence_type)
    for key in collocation_dict[sentence_type][feature].keys():
        if collocation_dict[sentence_type][feature][key] > min_freq:
            print("KEY: " + key + " -> VALUE: " + str(collocation_dict[sentence_type][feature][key]))
    print("------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('transcript_file', help='Directory containing transcription files.')
    args = vars(parser.parse_args())
    transcript_file = args['transcript_file']
    if not os.path.isfile(transcript_file):
        raise RuntimeError('The given file does not exist!')
    data = main.read_data(transcript_file)
    for sen in data:
        train_classifier(sen)
    print_common_features('GOOD_END', 'first_word', 500)
    print_common_features('GOOD_END', 'last_word', 500)
    print_common_features('GOOD_END', 'sentence_length', 500)

    print_common_features('BAD_END', 'first_word', 500)
    print_common_features('BAD_END', 'last_word', 500)
    print_common_features('BAD_END', 'sentence_length', 500)

    print_common_features('GOOD_START', 'first_word', 500)
    print_common_features('GOOD_START', 'last_word', 500)
    print_common_features('GOOD_START', 'sentence_length', 500)

    print_common_features('BAD_START', 'first_word', 500)
    print_common_features('BAD_START', 'last_word', 500)
    print_common_features('BAD_START', 'sentence_length', 500)