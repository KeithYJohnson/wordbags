import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from six.moves import cPickle as pickle
from ipdb import set_trace as st
from load_data import *
from params import *
from ipdb import set_trace as st

forest = load_data('forest.pickle')
cleaned_test_data = pd.read_csv(CLEAN_TEST_DATA, header=0)
test_data_features = vectorizer.fit_transform(cleaned_test_data['review'])
test_data_features = test_data_features.toarray()
result = forest.predict(test_data_features)

f = open(PREDICTION_PICKLE_FILE, 'wb')
pickle.dump(forest, f, pickle.HIGHEST_PROTOCOL)
f.close()

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":cleaned_test_data["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv("bagofwords.csv", index=False, quoting=3)
