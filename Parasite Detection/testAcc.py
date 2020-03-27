import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
import sklearn
from sklearn import metrics

""" Loading the dataset """ 
data1 = pd.read_csv("blob_features.csv")
data2 = pd.read_csv("contour_features.csv")
data3 = pd.read_csv("LUCID_features.csv")
data4 = pd.read_csv("num_of_contours.csv")
data1 = data1.iloc[:,1:]
data2 = data2.iloc[:,1:]
data3 = data3.iloc[:,1:]
data4 = data4.iloc[:,1:]
dfT = pd.DataFrame(data = np.concatenate([np.zeros(int(data1.shape[0]/2.)), np.ones(int(data1.shape[0]/2.))]))
data = pd.concat([dfT, data1, data2, data3, data4], axis = 1)

test1 = pd.read_csv("blob_features_test.csv")
test2 = pd.read_csv("contour_features_test.csv")
test3 = pd.read_csv("LUCID_features_test.csv")
test4 = pd.read_csv("num_of_contours_test.csv")
test1 = test1.iloc[:,1:]
test2 = test2.iloc[:,1:]
test3 = test3.iloc[:,1:]
test4 = test4.iloc[:,1:]
dfT = pd.DataFrame(data = np.concatenate([np.zeros(int(test1.shape[0]/2.)), np.ones(int(test1.shape[0]/2.))]))
test = pd.concat([dfT, test1, test2, test3, test4], axis = 1)

""" Building the train_set and the test_set """

# seperating the training and validation test
X_train = data.iloc[:,1:]
y_train = data.iloc[:,0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

acc = 0
for i in range(10):

# feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print("------Scaled the features--------")
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    model = RandomForestClassifier(n_estimators=75,max_depth=10)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score
    x = accuracy_score(y_pred, y_test)
    print(x)
    acc = acc + x

print("Accuracy -> ", acc*10, "%")

