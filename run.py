import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from six.moves import cPickle as pickle
from load_data import *
from params import *


train = pd.read_csv("cleanedlabledTrainData.tsv", header=0, quoting=3)

print('train.shape: ', train.shape)
print('train.columns.values: ', train.columns.values)
print('sampling first five reviews from {} total reviews'.format(train['review'].size))
print(train["review"][:5])

train_data_features = vectorizer.fit_transform(train['review'])
vocab = vectorizer.get_feature_names()
print('sampling vocab: ', vocab[0:20])

forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
forest = forest.fit( train_data_features, train["sentiment"] )

f = open(FOREST_PICKLE_FILE, 'wb')
pickle.dump(forest, f, pickle.HIGHEST_PROTOCOL)
f.close()

test = pd.read_csv(CLEAN_TEST_DATA, header=0, quoting=3)
test_data_features = vectorizer.transform(clean_test_reviews['review'])
test_data_features = test_data_features.toarray()
result = forest.predict(test_data_features)


print('test.shape: ', test.shape)
