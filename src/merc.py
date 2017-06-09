import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import  LabelEncoder


#We read the datasets
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
print(train['X0'])