"""
Author: Alex Hedges
Written for Python 3.6.3.
"""

import argparse
import csv
import os
import random

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

def main():
	"""Reads in transcript data and tests the turn-taking detector"""
	parser = argparse.ArgumentParser(description=
			'Runs turn-taking detector.')
	parser.add_argument('transcript_file', help='File containing the collated utterance information.')
	args = vars(parser.parse_args())
	transcript_file = args['transcript_file']
	if not os.path.isfile(transcript_file):
		raise RuntimeError('The given file does not exist!')

	random.seed(1311)

	data = read_data(transcript_file)
	sentences = [sentence for file_id, turn_type, speaker, turn_num, utt_num, sentence,
		good_start, good_end in data]
	random.shuffle(sentences)
	print(sentences[:10])

if __name__ == '__main__':
	main()
