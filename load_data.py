from six.moves import cPickle as pickle

def load_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickled_data = pickle.load(f)

    return pickled_data
