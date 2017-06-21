from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
#tf.logging.set_verbosity(tf.logging.INFO)

#Learning rate for the model
LEARNING_RATE = 0.01
dropout = 0.2

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
    # Load datasets
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

    xtest = test.as_matrix()
    # Scale data
    scaler = preprocessing.MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.fit_transform(xtest)

    # Set model params
    model_params = {"learning_rate": LEARNING_RATE,
    				"dropout":dropout}

    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    nn = tf.contrib.learn.Estimator(
        model_fn=model_fn, params=model_params)
    # Fit

    #Split data
    #xtest = xtrain[4000:]
    #ytest = ytrain[4000:]
    #xtrain = xtrain[:4000]
    #ytrain = ytrain[:4000]
    nn.fit(x=xtrain, y=ytrain, steps=200000)

    # Print out predictions
    predictions = nn.predict(x=xtest,
                             as_iterable=True)
    with open('resultsJiwo.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["ID", "y"])
        for i, p in enumerate(predictions):
            writer.writerow([labels[i], p['y']])


def model_fn(features, targets, mode, params):
    """Model function for Estimator."""

    # Connect the first hidden layer to input layer
    # (features) with relu activation
    first_hidden_layer = tf.contrib.layers.relu(features, 385)

    drop_out1 = tf.nn.dropout(first_hidden_layer, params['dropout'])  # DROP-OUT here
    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.contrib.layers.relu(drop_out1, 250)

    drop_out2 = tf.nn.dropout(second_hidden_layer, params['dropout'])

    third_hidden_layer = tf.contrib.layers.relu(drop_out2, 100)

    drop_out3 = tf.nn.dropout(third_hidden_layer, params['dropout'])

    fourth_hidden_layer = tf.contrib.layers.relu(drop_out3, 10)

    drop_out4 = tf.nn.dropout(fourth_hidden_layer, params['dropout'])

    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.contrib.layers.linear(drop_out4, 1)

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])
    predictions_dict = {"y": predictions}

    # Calculate loss using mean squared error
    loss = tf.contrib.losses.mean_squared_error(predictions, targets)
    train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer="SGD")

    return predictions_dict, loss, train_op
if __name__ == "__main__":
  tf.app.run()