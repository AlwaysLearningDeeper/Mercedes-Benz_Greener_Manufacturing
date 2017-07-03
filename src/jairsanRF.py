

import sys
import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_random_state

random_state=17
random_state = check_random_state(random_state)

#Learning rate for the model

def main():
    # Load datasets
    tf.set_random_seed(17)
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    categorical = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]

    # Remove ID, store it in separate variable in case of test data
    train = train.drop("ID", axis=1)
    labels = test["ID"]
    test = test.drop("ID", axis=1)

    # Transform categorical data
    for l in categorical:
        ec = preprocessing.LabelEncoder()
        ec.fit(list(train[l].values) + list(test[l].values))
        train[l] = ec.transform(list(train[l].values))
        test[l] = ec.transform(list(test[l].values))
        train[l] = train[l].astype(float)
        test[l] = test[l].astype(float)

    ytrain = train["y"].as_matrix()
    train = train.drop("y", axis=1)

    xtrain = train.as_matrix()


    # Split data
    xdev = xtrain[3800:]
    ydev = ytrain[3800:]
    xtrain = xtrain[:3800]
    ytrain = ytrain[:3800]

    regressor = RandomForestRegressor(n_estimators=100,max_depth=3,verbose=True,random_state=random_state)
    regressor.fit(xtrain, ytrain)




    predictions = regressor.predict(test.as_matrix())

    score = regressor.score(xdev, ydev)
    print(score)

    with open('resultsRFJairsan_5.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["ID", "y"])
        for i in range(0,len(predictions)):
            writer.writerow([labels[i], predictions[i]])

if __name__ == "__main__":
  main()