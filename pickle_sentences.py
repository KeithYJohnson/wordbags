from review_to_sentences import *
from params import *
import pandas as pd
import nltk.data
from six.moves import cPickle as pickle

labeled_train = pd.read_csv('cleaned_rmswFalse_rmnlFalselabeledTrainData.tsv', header=0)
unlabled_tarin = pd.read_csv('cleaned_rmswFalse_rmnlFalseunlabeledTrainData.tsv', header=0)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = []

print("Parsing sentences from training set")
for review in labeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

f = open(SENTENCES_PICKLE_FILE, 'wb')
pickle.dump(sentences, f, pickle.HIGHEST_PROTOCOL)
f.close()
