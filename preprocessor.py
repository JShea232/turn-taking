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
import re

PUNK_REGEX = re.compile(r'[^\w\-{<> /\'\*\[\]]')
NONALPHA_REGEX = re.compile(r'[^\w\- \']')

def store_data(file_name: str, data: list) -> None:
	"""Stores data in data file.

	Args:
		file_name: The name of the file to write to.
		data: List of data to store
	"""
	with open(file_name, 'w', encoding='utf8') as data_file:
		writer = csv.writer(data_file, delimiter='\t', lineterminator='\n',
			quotechar='', quoting=csv.QUOTE_NONE)
		writer.writerows(data)

def remove_enclosing(sentence: str, invalid_enclosings: list) -> str:
	"""Removes enclosed portions of a string.

	Args:
		sentence: The string to process.
		invalid_enclosings: A list of ordered pairs of enclosing strings.

	Returns:
		A string without the specified enclosed portions.
	"""
	count = 0
	for invalid_enclosing in invalid_enclosings:
		while sentence.find(invalid_enclosing[0]) != -1: # Remove all bracketed areas
			start = sentence.find(invalid_enclosing[0])
			end = sentence.find(invalid_enclosing[1]) + len(invalid_enclosing[1])
			if end == -1:
				raise RuntimeError('No "{}" found after "{}"!'.format(
					invalid_enclosing[1], invalid_enclosing[0]))
			bracketed = sentence[start:end]
			sentence = sentence.replace(bracketed, '', 1)
			count += 1
			if count > 15:
				exit()
	return sentence

def remove_tokens(sentence: str, invalid_starts: list, invalid_ends: list) -> str:
	"""Removes certain tokens from a string.

	Args:
		sentence: The string to process.
		invalid_starts: A list of starting strings for tokens to remove.
		invalid_ends: A list of ending strings for tokens to remove.

	Returns:
		A string without the specified tokens.
	"""
	tokens = sentence.split()
	for invalid_start in invalid_starts:
		tokens = [token for token in tokens if not token.startswith(invalid_start)]
	for invalid_end in invalid_ends:
		tokens = [token for token in tokens if not token.endswith(invalid_end)]
	sentence = ' '.join(tokens)
	return sentence

def collate_call_home(directory_name: str) -> list:
	"""Parses CallHome transcripts.

	This function is not implemented and raises an error when called.

	Args:
		directory_name: Name of the directory containing the transcripts.

	Returns:
		A list of utterances.
	"""
	raise NotImplementedError('collate_call_home() method not implemented')
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
				sentence = remove_tokens(sentence, ['&', '%'], [])
				print(sentence.encode('utf-8'))
				sentence = remove_enclosing(sentence, [('[', ']')])
				print(sentence.encode('utf-8'))
				utterances.append([file_id, prefix, line_number, sentence])
		#break
	return utterances

def process_switchboard_utterance(line: str, file_id: str) -> str:
	"""Parses Switchboard transcript line.

	Both utterance information and the sentence itself is extracted, as well as some turn-taking
	information. The sentences are processed to remove all annotations and normalizes them to a
	string of space-delimited word tokens.

	Args:
		line: Line for an utterance from the transcript file.
		file_id: ID of the transcript file.

	Returns:
		A list containing utterance information.
	"""
	row_info = line[:line.find(':')].strip().split()
	turn_type = row_info[0]
	speaker, turn_num = row_info[1].split('.')
	utt_num = row_info[2][3:]
	sentence = line[line.find(':') + 1:].strip()
	good_end = sentence[-1:] == '/' and not sentence[-2:] == '-/'
	good_start = not sentence[:2] == '--'
	sentence = PUNK_REGEX.sub('', sentence) # Remove most punctuation
	sentence = remove_tokens(sentence, ['{'], ['-']) # Remove some tokens with invalid characters
	sentence = remove_enclosing(sentence, [('<<', '>>'), ('<', '>'), ('*[[', ']]')]) # Remove enclosed text
	
	sentence = NONALPHA_REGEX.sub('', sentence) # Remove remaining punctuation and lowercase
	sentence = ' '.join([token for token in sentence.split() if token != '-']) # Remove '-' tokens
	return [file_id, turn_type, speaker, turn_num, utt_num, sentence, good_start, good_end]

def collate_switchboard(directory_name: str) -> list:
	"""Parses Switchboard transcripts.

	This function finds all utterance files within the given directory and its subdirectories and
	reads them into the program. Both utterance information and the sentence itself is extracted,
	as well as some turn-taking information. The sentences are processed to remove all annotations
	and normalizes them to a string of space-delimited word tokens.

	Args:
		directory_name: Name of the directory containing the transcripts.

	Returns:
		A list of utterances.
	"""
	files = [] # Find all files recursively in directory
	for root, _, dir_files in os.walk(directory_name):
		for file in dir_files:
			files.append(join(root, file))
	files = sorted(files)
	utterances = []
	for file_name in files:
		if splitext(basename(file_name))[1] != '.utt': # Skip file if not a transcript
			continue
		file_id = splitext(basename(file_name))[0]
		print('Extracting from {}...'.format(file_id), flush=True)
		with open(file_name, 'r', encoding='utf8') as file:
			hit_starting_line = False
			for line in file:
				if line and line[0] == '=': # Ignore header and blank lines
					hit_starting_line = True
					continue
				if not hit_starting_line or not line.strip():
					continue
				utterances.append(process_switchboard_utterance(line, file_id))
	return utterances

def main():
	"""Collates transcripts from the given directory and outputs them to a TSV file"""
	parser = argparse.ArgumentParser(description=
			'Collates directory of transcription files into a single file.')
	parser.add_argument('transcript_directory', help='Directory containing transcription files.')
	parser.add_argument('output_file', help='File to store collated transcripts.')
	args = vars(parser.parse_args())
	transcript_directory = args['transcript_directory']
	output_file = args['output_file']
	if not os.path.isdir(transcript_directory):
		raise RuntimeError('The given directory "{}" does not exist!'.format(transcript_directory))

	data = collate_switchboard(transcript_directory)
	store_data(output_file, data)

if __name__ == '__main__':
	main()
