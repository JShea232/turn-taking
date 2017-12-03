"""
Author: Alex Hedges
Written for Python 3.6.3.
Requires NLTK and its subpackage averaged_perceptron_tagger
"""

import argparse
from itertools import accumulate
import os
import random
import re
import string

import nltk
from nltk import pos_tag
from nltk.grammar import Nonterminal
from nltk.grammar import PCFG
from nltk.grammar import Production
from nltk.parse import ViterbiParser

from main import read_data

REMOVE_DIGITS_REGEX = re.compile(r'^(.*?)([=\-]\d+)*$')
REMOVE_SUFFIX_REGEX = re.compile(r'^(.*?)((?:-|=).*)?$')

def simplify_nonterminal(nonterm: Nonterminal) -> Nonterminal:
	return Nonterminal(REMOVE_SUFFIX_REGEX.fullmatch(nonterm.symbol()).group(1))

def valid_nonterminal(nonterm: Nonterminal) -> bool:
	return nonterm.symbol()[0] in string.ascii_letters

def create_grammar() -> PCFG:
	# 21,763 productions with word terminals
	# 8,028 productions with pos terminals
	# 6,275 productions with nonterminals without digits
	# 5,402 productions with nonterminals without punctuation
	# 2,972 productions with nonterminals without suffixes
	# 707 nonterminals
	# 190 nonterminals without digit labels
	# 180 nonterminals without punctuation
	# 63 nonterminals without suffixes
	productions = []
	start_symbol = Nonterminal('S')
	for tree in nltk.corpus.treebank.parsed_sents():
		for production in tree.productions():
			if not valid_nonterminal(production.lhs()):
				continue
			if isinstance(production.rhs()[0], Nonterminal):
				lhs = simplify_nonterminal(production.lhs())
				rhs = tuple(simplify_nonterminal(t) for t in production.rhs() if valid_nonterminal(t))
				productions.append(Production(lhs, rhs))
			else:
				simplified = simplify_nonterminal(production.lhs())
				productions.append(Production(simplified, (simplified.symbol(),)))

	grammar = nltk.induce_pcfg(start_symbol, productions)
	#print(grammar.productions())
	print(len(grammar.productions()))
	nonterminals = set(prod.lhs() for prod in grammar.productions())
	print(sorted(nonterminals))
	print(len(nonterminals))
	return grammar

def evaluate_sentence(sentence: string, grammar: PCFG):
	sentence = sentence.split()
	print(sentence, flush=True)
	pos = [pos for word, pos in pos_tag(sentence)]
	print(pos, flush=True)
	parser = ViterbiParser(grammar, trace=0)
	for line in accumulate(pos, lambda total, token: total + ' ' + token):
		line = line.split()
		print(line)
		print([tree.prob() for tree in list(parser.parse(line))], flush=True)

def main():
	"""Tests the syntax analyzing portions of the turn-taking detector"""
	parser = argparse.ArgumentParser(description='Runs turn-taking detector.')
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

	grammar = create_grammar()
	sentence = 'that\'s where the real sticking point comes in'
	# "I went to the house"
	# "The house I did go"
	# "that's where the real sticking point comes in"
	evaluate_sentence(sentence, grammar)

if __name__ == '__main__':
	main()
