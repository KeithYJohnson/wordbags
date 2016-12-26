from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from params import WORD2VEC_MODEL, KMEANS_PICKLE, KMEANS_DICT
from six.moves import cPickle as pickle
import time

start = time.time()
model = Word2Vec.load(WORD2VEC_MODEL)
word_vectors = model.syn0
# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
num_clusters = int(word_vectors.shape[0] / 5)

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

f = open(KMEANS_PICKLE, 'wb')
pickle.dump(idx, f, pickle.HIGHEST_PROTOCOL)
f.close()

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start

# Local Time taken for K Means clustering:  486.05815601348877 seconds.
print("Time taken for K Means clustering: ", elapsed, "seconds.")

word_centroid_map = dict(zip( model.index2word, idx ))
f = open(KMEANS_DICT, 'wb')
pickle.dump(word_centroid_map, f, pickle.HIGHEST_PROTOCOL)
f.close()
