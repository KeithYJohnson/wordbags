from sklearn.feature_extraction.text import CountVectorizer
FOREST_PICKLE_FILE = 'forest.pickle'
PREDICTION_PICKLE_FILE = 'results.pickle'
CLEAN_TEST_DATA = 'cleaned_testData.tsv'

vectorizer = CountVectorizer(
    analyzer = "word",
    tokenizer = None,
    preprocessor = None,
    stop_words = None,
    max_features = 5000
)
