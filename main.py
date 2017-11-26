"""
Author: Alex Hedges
Written for Python 3.6.3.
"""

import argparse
import csv
import os
import random

import nltk
from nltk import pos_tag

def read_data(file_name: str) -> list:
	data = []
	with open(file_name, 'r', encoding='utf8') as data_file:
		reader = csv.reader(data_file, delimiter='\t', lineterminator='\n',
			quotechar='', quoting=csv.QUOTE_NONE)
		for row in reader:
			file_id = row[0]
			turn_type = row[1]
			speaker = row[2]
			turn_num = row[3]
			utt_num = row[4]
			sentence = row[5]
			good_start = row[6]
			good_end = row[7]
			if good_start and good_end:
				data.append([file_id, turn_type, speaker, turn_num, utt_num, sentence])
	return data

def main():
	parser = argparse.ArgumentParser(description=
			'Collates directory of transcription files into a signle file.')
	parser.add_argument('transcript_file', help='Directory containing transcription files.')
	args = vars(parser.parse_args())
	transcript_file = args['transcript_file']
	if not os.path.isfile(transcript_file):
		raise RuntimeError('The given file does not exist!')

	random.seed(1311)
		
	data = read_data(transcript_file)
	sentences = [sentence for file_id, turn_type, speaker, turn_num, utt_num, sentence in data]
	random.shuffle(sentences)
	print(sentences[:10])
	
	productions = []
	S = nltk.Nonterminal('S')
	for tree in nltk.corpus.treebank.parsed_sents():
		#print(tree.productions())
		for production in tree.productions():
			if type(production.rhs()[0]) == nltk.grammar.Nonterminal:
				productions.append(production)
			else:
				productions.append(nltk.grammar.Production(production.lhs(), (production.lhs().symbol(),)))
		#print(productions)
		#exit()
	grammar = nltk.induce_pcfg(S, productions)
	#print(grammar.productions()[1000:1020])
	#print(grammar.check_coverage(['NN', 'VBP']))
	sentence = 'I went to the house'.split() # The house I did go
	print(sentence, flush=True)
	pos = [pos for word, pos in pos_tag(sentence)]
	print(pos, flush=True)
	#for 
	print([tree.prob() for tree in list(nltk.parse.ViterbiParser(grammar, trace=0).parse(pos))], flush=True)

if __name__ == '__main__':
	main()
