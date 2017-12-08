"""
Author: Alex Hedges
Written for Python 3.6.3.
"""

import argparse
import csv
import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from baseline import BaselineClassifier
from tfidf import TfidfClassifier

def read_data(file_name: str) -> list:
	"""Reads in a TSV file and converts to a list of utterances.

	Args:
		file_name: The name of the file to read in.

	Returns:
		A list of rows, each of which is the data for an utterance.
	"""
	data = []
	with open(file_name, 'r', encoding='utf8') as data_file:
		reader = csv.reader(data_file, delimiter='\t', lineterminator='\n',
			quotechar='', quoting=csv.QUOTE_NONE)
		for row in reader:
			file_id = row[0]
			turn_type = row[1]
			speaker = row[2]
			turn_num = int(row[3])
			utt_num = int(row[4])
			sentence = row[5]
			good_start = row[6] == 'True'
			good_end = row[7] == 'True'
			if good_start:
				data.append([file_id, turn_type, speaker, turn_num, utt_num, sentence, good_start, good_end])
	return data

def evaluate(y_true: list, y_pred: np.ndarray) -> None:
	"""Calculates metrics for a model."""
	print('Accuracy: {:.4f}'.format(accuracy_score(y_true, y_pred)))
	print('Precision: {:.4f}'.format(precision_score(y_true, y_pred, average='macro')))
	print('Recall: {:.4f}'.format(recall_score(y_true, y_pred, average='macro')))
	print('F1: {:.4f}'.format(f1_score(y_true, y_pred, average='macro')))

def main():
	"""Reads in transcript data and tests the turn-taking detector"""
	parser = argparse.ArgumentParser(
		description='Trains and gives information for a turn-taking model.')
	parser.add_argument('transcript_file', help='File containing the collated '
												'utterance information.')
	parser.add_argument('-t', action='store_true', help='Enable training.', dest='train')
	parser.add_argument('-model', default='tfidf', choices=['baseline', 'tfidf'], help='Model to use.')
	args = vars(parser.parse_args())
	transcript_file = args['transcript_file']
	do_train = args['train']
	model_name = args['model']
	if not os.path.isfile(transcript_file):
		raise RuntimeError('The given file does not exist!')

	data = read_data(transcript_file)
	sentences, endings = zip(*[(sentence, good_end) for file_id, turn_type, speaker, turn_num,
		utt_num, sentence, good_start, good_end in data])
	train_sentences, test_sentences, train_endings, test_endings = train_test_split(
		sentences, endings, test_size=0.1, random_state=1311)

	if model_name == 'baseline':
		model = BaselineClassifier()
	elif model_name == 'tfidf':
		model = TfidfClassifier()

	if do_train:
		model.train_model(train_sentences, train_endings)

	predictions = model.test_model(test_sentences)

	evaluate(test_endings, predictions)

if __name__ == '__main__':
	main()
