from sklearn import preprocessing
import pickle
import pandas as pd
def save_object(object, file_name):
    with open(file_name, 'wb') as fh:
        pickle.dump(object, fh)
def main():
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
    save_object(xtrain,'xtrain')
    save_object(labels,'labels')
    save_object(ytrain,'ytrain')
    save_object(xtest,'xtest')
main()
