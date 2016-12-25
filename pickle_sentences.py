from review_to_sentences import *
from params import *
import pandas as pd
import nltk.data
from six.moves import cPickle as pickle

labeled_train = pd.read_csv('cleaned_rmswFalse_rmnlFalselabeledTrainData.tsv', header=0)
unlabeled_train = pd.read_csv('cleaned_rmswFalse_rmnlFalseunlabeledTrainData.tsv', header=0)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = []

print('sentencing {} labeled examples'.format(labeled_train.shape))
for idx, review in enumerate(labeled_train["review"]):
    print('labeled: ', idx)
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
print('sentencing {} unlabeled examples'.format(unlabeled_train.shape))
for idx, review in enumerate(unlabeled_train["review"]):
    print('unlabeled: ', idx)
    sentences += review_to_sentences(review, tokenizer)

f = open(SENTENCES_PICKLE_FILE, 'wb')
pickle.dump(sentences, f, pickle.HIGHEST_PROTOCOL)
f.close()
