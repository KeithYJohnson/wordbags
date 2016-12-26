import pandas as pd
import numpy as np
from params import CENTROIDS_BAG, KMEANS_DICT
from review_to_words import *
from sklearn.ensemble import RandomForestClassifier
from create_bag_of_centroids import *
from load_data import load_data

word_centroid_map = load_data(KMEANS_DICT)

num_clusters = 3298

train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_words( review, \
        remove_stopwords=True, asarray=True ))

test = pd.read_csv("testData.tsv",  header=0, delimiter="\t", quoting=3)


clean_test_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_words( review, \
        remove_stopwords=True, asarray=True))


# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (train["review"].size, num_clusters), \
    dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# Repeat for test reviews
test_centroids = np.zeros(( test["review"].size, num_clusters), \
    dtype="float32" )

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# Fit a random forest and extract predictions
forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)

f = open(CENTROIDS_BAG, 'wb')
pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
f.close()

# Write the test results
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )
