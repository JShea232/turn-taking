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
