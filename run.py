import pandas as pd

train = pd.read_csv("cleanedlabledTrainData.tsv", header=0, quoting=3)

print('train.shape: ', train.shape)
print('train.columns.values: ', train.columns.values)
print('sampling first five reviews from {} total reviews'.format(train['review'].size))
print(train["review"][:5])

train_data_features = vectorizer.fit_transform(train['review'])
vocab = vectorizer.get_feature_names()
print('sampling vocab: ', vocab[0:20])
