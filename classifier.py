"""
Author: Jordan Shea
Written for Python 3.6.3.
"""

import argparse
import os

from sklearn.ensemble import GradientBoostingClassifier
from main import read_data


def main():
    # Reads in transcript training data to build a Boosted Tree model
    parser = argparse.ArgumentParser(
        description='Trains and gives information for a Boosted-Tree model.')
    parser.add_argument('transcript_file', help='File containing the collated '
                                                'utterance information.')
    args = vars(parser.parse_args())
    transcript_file = args['transcript_file']
    if not os.path.isfile(transcript_file):
        raise RuntimeError('The given file does not exist!')
    data = read_data(transcript_file)


if __name__ == "__main__":
    main()
