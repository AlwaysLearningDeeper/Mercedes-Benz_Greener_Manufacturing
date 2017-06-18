from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
tf.logging.set_verbosity(tf.logging.INFO)

#Learning rate for the model
LEARNING_RATE = 0.001

#tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
    # Load datasets
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    categorical = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]

    # Remove ID, store it in separate variable in case of test data
    train = train.drop("ID", axis=1)
    ytest = test["ID"]
    xtest = test.drop("ID", axis=1)

    # Transform categorical data
    for l in categorical:
        ec = preprocessing.LabelEncoder()
        ec.fit(list(train[l].values) + list(xtest[l].values))
        train[l] = ec.transform(list(train[l].values))
        xtest[l] = ec.transform(list(xtest[l].values))
        train[l] = train[l].astype(float)
        xtest[l] = xtest[l].astype(float)

    ytrain = train["y"].as_matrix()
    train = train.drop("y", axis=1)

    xtrain = train.as_matrix()

    # Scale data
    scaler = preprocessing.MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)

    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}

    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    nn = tf.contrib.learn.Estimator(
        model_fn=model_fn, params=model_params)
    # Fit
    nn.fit(x=xtrain, y=ytrain, steps=5000)

    # Score accuracy
    ev = nn.evaluate(x=xtest, y=ytest, steps=1)
    loss_score = ev["loss"]
    print("Loss: %s" % loss_score)

    # Print out predictions
    predictions = nn.predict(x=ytest,
                             as_iterable=True)
    for i, p in enumerate(predictions):
        print("Prediction %s: %s" % (i + 1, p["ages"]))

def model_fn(features, targets, mode, params):
    """Model function for Estimator."""

    # Connect the first hidden layer to input layer
    # (features) with relu activation
    first_hidden_layer = tf.contrib.layers.relu(features, 10)

    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 10)

    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)

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