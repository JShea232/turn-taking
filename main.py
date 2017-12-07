"""
Author: Alex Hedges
Written for Python 3.6.3.
"""

import argparse
import csv
import os
import random

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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

def evaluate(model, x_data: list, y_data: list) -> (float, float, float, float):
	"""Calculates metrics for a model."""
	x_data = np.reshape(range(len(x_data)), (-1, 1))
	y_pred = model.predict(x_data)
	y_true = y_data

	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred, average='macro')
	recall = recall_score(y_true, y_pred, average='macro')
	f1_measure = f1_score(y_true, y_pred, average='macro')
	return accuracy, precision, recall, f1_measure

def train_test_split(features: list, targets: list,
	test_size: float, random_state: int = None) -> (list, list, list, list):
	"""Working version of sklearn.model_selection.train_test_split"""
	if random_state:
		random.seed(random_state)
	data = list(zip(features, targets))
	random.shuffle(data)
	features, targets = zip(*data)
	split = round(len(features) * (1 - test_size))
	x_train = features[:split]
	y_train = targets[:split]
	x_test = features[:split]
	y_test = targets[:split]
	return x_train, x_test, y_train, y_test

def main():
	"""Reads in transcript data and tests the turn-taking detector"""
	parser = argparse.ArgumentParser(description=
			'Runs turn-taking detector.')
	parser.add_argument('transcript_file', help='File containing the collated utterance information.')
	args = vars(parser.parse_args())
	transcript_file = args['transcript_file']
	if not os.path.isfile(transcript_file):
		raise RuntimeError('The given file does not exist!')

	data = read_data(transcript_file)
	sentences = [sentence for file_id, turn_type, speaker, turn_num, utt_num, sentence,
		good_start, good_end in data]
	good_ends = [good_end for file_id, turn_type, speaker, turn_num, utt_num, sentence,
		good_start, good_end in data]
	x_train, x_test, y_train, y_test = train_test_split(
		sentences, good_ends, test_size=0.1, random_state=1311)

	baseline = DummyClassifier(strategy='most_frequent')
	baseline.fit(np.reshape(range(len(x_train)), (-1, 1)), y_train)
	print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1: {}'.format(
		*evaluate(baseline, x_test, y_test)))

if __name__ == '__main__':
	main()
