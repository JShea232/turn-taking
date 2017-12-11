Automated Detection of Acceptable Turn-Taking in Conversation

Contributors:	Jordan Shea
		Alexander Hedges
Viewers:	Professor Cecilia Alm

In total, our program has 5 separate classes...
- baseline.py 		(Class for computing the baseline for a test set)
- grammar.py 		(Scrapped idea for constructing a custom grammar)
- main.py 		(Main entry point for all the sub-programs)
- preprocessor.py 	(Class for stripping and pre-processing the text)
- tfidf.py 		(Class for implementing our SVM and Decision Tree models)
- yarowsky.py 		(Class for implementing the Yarowsky algorithm on our data)

In order to run our code, here are some of the dependencies that you'll need 
to have installed (each of these can be installed via pip)
- matplotlib == 2.1.0
- nltk == 3.2.5
- sklearn == 0.0
- scikit-learn == 0.19.1

In order to run the base program to test our models on a data set, do...
- python main.py dialogue.tsv

In order to run the base program to test AND train our models on a data set, do...
- python main.py dialogue.tsv -t

In order to calculate the baseline of the test set, do...
- python main.py dialogue.tsv -model baseline

In order to run our Yarowsky classifier and see the most characteristic features, do...
- python yarowsky.py dialogue.tsv

For your educational purposes, we included the grammar.py class, which was one of our attempts
at creating a classifier by constructing our own custom grammar. While we inevitably scrapped this
idea, you can still view the code that was currently in progress.

For future work, we would like to expand upon this code by being able to...
1) classify using only yarowsky.py, and seeing how these results compare to our baseline
2) augment the parameters of our models within our tfidf.py class 
