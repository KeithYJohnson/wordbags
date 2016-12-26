#### To play around with the model.

It's a large file so I didn't commit it.  To train your own:

1.  Run `python3 clean_data.py`
2.  Run `python3 pickle_sentences.py`
3.  run `python3 train_word2vec.py`

```python
from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context")

model.doesnt_match("man woman child kitchen".split())
model.doesnt_match("france england germany berlin".split())
model.doesnt_match("paris berlin london austria".split())
model.most_similar("man")
model.most_similar("awful")
```
