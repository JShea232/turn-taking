"""
Author: Alex Hedges
Written for Python 3.6.3.
"""

import argparse
import csv
import os

import pandas as pd

def read_data(file: str) -> pd.DataFrame:
	df = pd.read_csv(file, quoting=csv.QUOTE_NONE, sep='\t', header=None)

	df.columns = ['file_id', 'turn_type', 'speaker', 'turn_num', 'utt_num', 'sentence', 'good_start', 'good_end']
	#df = df.set_index(['file_id', 'turn_num', 'utt_num'])

	df['turn_type'] = df['turn_type'].astype('category')
	df = df.drop(columns=['turn_type'])	

	return df

def main():
	parser = argparse.ArgumentParser(description=
			'Collates directory of transcription files into a signle file.')
	parser.add_argument('transcript_file', help='Directory containing transcription files.')
	args = vars(parser.parse_args())
	transcript_file = args['transcript_file']
	if not os.path.isfile(transcript_file):
		raise RuntimeError('The given file does not exist!')

	data = read_data(transcript_file)
	data = data[data.good_start & data.good_end]
	data = data.drop(columns=['good_start', 'good_end'])
	sentences = data['sentence'].sample(frac=1, random_state=1311)
	#print(sentences.head())

if __name__ == '__main__':
	main()
