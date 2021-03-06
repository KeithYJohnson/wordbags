from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_words(raw_review, remove_stopwords=True,  remove_nonletters=True, asarray=False):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, 'html5lib').get_text()
    #
    # 2. Remove non-letters
    if remove_nonletters:
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = review_text.lower().split()
    #
    if remove_stopwords:
        # 4. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))
        #
        # 5. Remove stop words
        words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    if asarray:
        return(words)
    else:
        return( " ".join( words ))
