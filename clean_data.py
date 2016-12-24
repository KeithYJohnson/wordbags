from review_to_words import *
import pandas as pd

def clean_data(filename, column_to_clean='review'):
    data = pd.read_csv(filename, header=0, delimiter="\t", quoting=3 )

    data[column_to_clean].map(lambda x: review_to_words(x))
    data.to_csv('cleaned_' + filename)

clean_data('testData.tsv')
