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
LEARNING_RATE = 0.0001
dropout = 0.2

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(17)

def main(unused_argv):
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

    xtest = test.as_matrix()
    # Scale data
    scaler = preprocessing.MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.fit_transform(xtest)

    # Split data
    xdev = xtrain[3800:]
    ydev = ytrain[3800:]
    xtrain = xtrain[:3800]
    ytrain = ytrain[:3800]


    # Set model params
    model_params = {"learning_rate": LEARNING_RATE,
    				"dropout":dropout}

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(xdev,ydev,early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=100)

    nn = tf.contrib.learn.SKCompat(tf.contrib.learn.Estimator(
        model_fn=model_fn, params=model_params,model_dir="./tmp/mercedes_model",
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=0.25,tf_random_seed=17)))


    # Fit

    nn.fit(x=xtrain, y=ytrain, batch_size = 40,steps=200,monitors=[validation_monitor])

    # Print out predictions
    predictions = nn.predict(x=xtest)["y"]


    with open('resultsJairsan_6.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["ID", "y"])
        for i in range(0,len(predictions)):
            writer.writerow([labels[i], predictions[i]])


def model_fn(features, targets, mode, params):
    """Model function for Estimator."""

    # Connect the first hidden layer to input layer
    # (features) with relu activation
    first_hidden_layer = tf.contrib.layers.fully_connected(features,175,activation_fn=tf.nn.relu)

    second_hidden_layer = tf.contrib.layers.fully_connected(first_hidden_layer, 50,activation_fn=tf.nn.relu)

    third_hidden_layer = tf.contrib.layers.fully_connected(second_hidden_layer, 5, activation_fn=tf.nn.relu)


    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.contrib.layers.linear(third_hidden_layer, 1)

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])
    predictions_dict = {"y": predictions}

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(predictions, targets)
    train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer="SGD")

    return predictions_dict, loss, train_op
if __name__ == "__main__":
  tf.app.run()