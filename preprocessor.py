"""
Author: Alex Hedges
Written for Python 3.6.3.
"""

import argparse
import csv
import os
from os.path import basename
from os.path import splitext
from os.path import join

def store_data(file_name: str, data: list) -> None:
	"""Stores data in data file.

	Args:
	  file_name: The name of the file to read from.
	  data: List of data to store

	"""
	with open(file_name, 'w', encoding='utf8') as data_file:
		writer = csv.writer(data_file, delimiter='\t', lineterminator='\n', quotechar='"')#, quoting=csv.QUOTE_NONE)
		writer.writerows(data)

def remove_delimiters(sentence: str, invalid_delimiters: list) -> str:
	for invalid_delimiter in invalid_delimiters:
		while sentence.find(invalid_delimiter[0]) != -1: # Remove all bracketed areas
			start = sentence.find(invalid_delimiter[0])
			end = sentence.find(invalid_delimiter[1])
			if end == -1:
				raise RuntimeError('No "{}" found after "{}"!'.format(invalid_delimiter[1], invalid_delimiter[0]))
			bracketed = sentence[start:end + 1]
			sentence = sentence.replace(bracketed, '', 1)
	return sentence
		
def remove_tokens(sentence: str, invalid_starts: list) -> str:
	for invalid_start in invalid_starts:
		sentence = ' '.join([token for token in sentence.split() if token[0] != invalid_start])
	return sentence
		
def collate_call_home(directory_name: str) -> list:
	files = sorted(list(set(os.listdir(directory_name))))
	utterances = []
	for file_name in files:
		print(file_name)
		if splitext(basename(file_name))[1] != '.cha':
			continue
		file_id = splitext(basename(file_name))[0]
		with open(join(directory_name, file_name), 'r', encoding='utf8') as file:
			line_number = 0
			for line in file:
				if not line or line[0] != '*':
					continue
				line_number += 1
				line = line.strip()
				prefix, sentence = line.split('\t')
				prefix = prefix[1:-1]
				if sentence.find('') != -1:
					sentence = sentence[:sentence.find('')]
				print(sentence.encode('utf-8'))
				sentence = remove_tokens(sentence, ['&', '%'])
				print(sentence.encode('utf-8'))
				sentence = remove_delimiters(sentence, [('[', ']')])
				print(sentence.encode('utf-8'))
				utterances.append([file_id, prefix, line_number, sentence])
		#break
	return utterances

def collate_switchboard(directory_name: str) -> list:
	files = sorted(list(set(os.listdir(directory_name))))
	utterances = []
	for file_name in files:
		print(file_name)
		if splitext(basename(file_name))[1] != '.utt':
			continue
		file_id = splitext(basename(file_name))[0]
		with open(join(directory_name, file_name), 'r', encoding='utf8') as file:
			hit_starting_line = False
			for line in file:
				if line and line[0] == '=':
					hit_starting_line = True
					continue
				if not hit_starting_line or not line.strip():
					continue
				
				
				row = line[:line.find(':')].strip().split()
				turn_type = row[0]
				speaker, turn_num = row[1].split('.')
				utt_num = row[2][3:]
				sentence = line[line.find(':') + 1:].strip()
				sentence = remove_tokens(sentence, ['{', '}'])
				print([file_id, turn_type, speaker, turn_num, utt_num, sentence])
				continue
				utterances.append([file_id, prefix, line_number, sentence])
		break
	return utterances

def main():
	parser = argparse.ArgumentParser(description=
			'Collates directory of transcription files into a signle file.')
	parser.add_argument('transcript_directory', help='Directory containing transcription files.')
	parser.add_argument('output_file', help='File to store collated transcripts.')
	args = vars(parser.parse_args())
	transcript_directory = args['transcript_directory']
	output_file = args['output_file']
	if not os.path.isdir(transcript_directory):
		raise RuntimeError('The given directory does not exist!')
	data = collate_switchboard(transcript_directory)
	store_data(output_file, data)

if __name__ == '__main__':
	main()
