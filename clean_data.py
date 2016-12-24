from review_to_words import *
import pandas as pd
from ipdb import set_trace as st

def clean_data(filename, column_to_clean='review', remove_stopwords=True, remove_nonletters=True):
    data = pd.read_csv(filename, header=0, delimiter="\t", quoting=3 )

    data[column_to_clean].map(
        lambda x: review_to_words(
            x, remove_stopwords=remove_stopwords, remove_nonletters=remove_nonletters
        )
    )

    data.to_csv('cleaned_rmsw{}_rmnl{}'.format(remove_stopwords, remove_nonletters) + filename)

# clean_data('testData.tsv')
clean_data('labeledTrainData.tsv',   remove_stopwords=False, remove_nonletters=False)
clean_data('unlabeledTrainData.tsv', remove_stopwords=False, remove_nonletters=False)
