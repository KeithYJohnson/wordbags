from sklearn.feature_extraction.text import CountVectorizer
FOREST_PICKLE_FILE = 'forest.pickle'
PREDICTION_PICKLE_FILE = 'results.pickle'
CLEAN_TEST_DATA = 'cleaned_testData.tsv'
SENTENCES_PICKLE_FILE = 'sentences.pickle'
WORD2VEC_MODEL = "300features_40minwords_10context"
KMEANS_PICKLE = 'kmeans.pickle'
KMEANS_DICT = 'kmeans_dict.pickle'
CENTROIDS_BAG = 'centroids_bag_model.pickle'

vectorizer = CountVectorizer(
    analyzer = "word",
    tokenizer = None,
    preprocessor = None,
    stop_words = None,
    max_features = 5000
)
