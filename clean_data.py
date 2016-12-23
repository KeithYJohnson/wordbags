from review_to_words import *
import pandas as pd

train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

train['review'] = train['review'].map(lambda x: review_to_words(x))

train.to_csv('cleanedlabledTrainData.tsv')
