"""
Author: Alex Hedges
Written for Python 3.6.3.
"""

import argparse
import csv
import itertools
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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

def plot_confusion_matrix(true: list, preds: np.ndarray, classes: list, normalize: bool = False,
	title: str = 'Confusion matrix',
	color: mpl.colors.LinearSegmentedColormap = plt.get_cmap('Blues')):
	"""This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.

	Source: http://scikit-learn.org/dev/_downloads/plot_confusion_matrix.py
	"""
	results = confusion_matrix(true, preds)

	if normalize:
		results = results.astype('float') / results.sum(axis=1)[:, np.newaxis]
		print('Normalized confusion matrix')
	else:
		print('Confusion matrix, without normalization')

	print(results)

	plt.imshow(results, interpolation='nearest', cmap=color)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = results.max() / 2
	for i, j in itertools.product(range(results.shape[0]), range(results.shape[1])):
		plt.text(j, i, format(results[i, j], fmt), horizontalalignment='center',
			color='white' if results[i, j] > thresh else 'black')

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	#plt.savefig('confusion_matrix.png', bbox_inches='tight')
	plt.show()

def evaluate(y_true: list, y_pred: np.ndarray) -> None:
	"""Calculates metrics for a model."""
	print('Accuracy: {:.4f}'.format(accuracy_score(y_true, y_pred)))
	print('Precision: {:.4f}'.format(precision_score(y_true, y_pred, average='macro')))
	print('Recall: {:.4f}'.format(recall_score(y_true, y_pred, average='macro')))
	print('F1: {:.4f}'.format(f1_score(y_true, y_pred, average='macro')))
	plot_confusion_matrix(y_true, y_pred, ['interrupted', 'uninterrupted'], normalize=True)

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
