import pandas as pd
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup as bs
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)


print('train.shape: ', train.shape)
print('train.columns.values: ', train.columns.values)
print('sampling first five reviews from {} total reviews'.format(train['review'].size))
print(train["review"][:5])






#Remove HTML Tags
