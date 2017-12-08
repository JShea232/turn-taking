"""
Author: Jordan Shea and Alex Hedges
Written for Python 3.6.3.
"""

import pickle

import numpy as np
from sklearn.dummy import DummyClassifier


class BaselineClassifier(object):
	@staticmethod
	def train_model(sentence_list: list, endings_list: list):
		"""
		This function works to train a model utilizing a Dummy Classifier
		By using this Dummy Classifier, it is possible to calculate a
		baseline by simply casting everything as the majority class.

		:param sentence_list: list of sample utterances
		:param endings_list: list of whether or not an utterance was interrupted
		:return: None
		"""

		baseline = DummyClassifier(strategy='most_frequent')
		baseline.fit(np.reshape(range(len(sentence_list)), (-1, 1)), endings_list)

		# Store model and vectorizer into a pickle file
		with open('baseline.pkl', 'wb') as file:
			pickle.dump(baseline, file)

	@staticmethod
	def test_model(sentences: list) -> np.ndarray:
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
		with open('baseline.pkl', 'rb') as file:
			baseline = pickle.load(file)

		x_data = np.reshape(range(len(sentences)), (-1, 1))
		predictions = baseline.predict(x_data)
		return predictions
