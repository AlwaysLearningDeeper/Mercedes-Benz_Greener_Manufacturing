import pandas as pd
import csv
from sklearn.neighbors import KNeighborsRegressor

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

categorical=["X0","X1","X2","X3","X4","X5","X6","X8"]

#Remove ID and store it for test results
train=train.drop("ID",axis=1)
labels=test["ID"]
test=test.drop("ID",axis=1)

#Drop categorical data
for l in categorical:
    train=train.drop(l,axis=1)
    test=test.drop(l,axis=1)

#Create dev set
i=int(train.shape[0]*0.9)
dev=train.iloc[i:-1]
train=train.iloc[:i]


ytrain=train["y"].as_matrix()
train=train.drop("y",axis=1)
xtrain=train.as_matrix()


regressor= KNeighborsRegressor(10,weights='distance')
regressor.fit(xtrain,ytrain)

actual=dev["y"].as_matrix()

dev=dev.drop("y",axis=1)
xdev=dev.as_matrix()


score=regressor.score(xdev,actual)

predictions=regressor.predict(test.as_matrix())

with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["ID","y"])
    for i in range(0,len(predictions)):
        writer.writerow([labels[i],predictions[i]])


