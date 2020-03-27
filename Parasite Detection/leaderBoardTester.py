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
data3 = pd.read_csv("num_of_contours.csv")
data4 = pd.read_csv("sift_features.csv")
data1 = data1.iloc[:,1:]
data2 = data2.iloc[:,1:]
data3 = data3.iloc[:,1:]
data4 = data4.iloc[:,1:]
dfT = pd.DataFrame(data = np.concatenate([np.zeros(int(data1.shape[0]/2.)), np.ones(int(data1.shape[0]/2.))]))
data = pd.concat([dfT, data1, data2, data3, data4], axis = 1)

test = pd.read_csv("test_features.csv")
test = test.iloc[:,1:]

""" Building the train_set and the test_set """ 

# seperating the training and validation test
X = data.iloc[:,1:]
y = data.iloc[:,0]
testX = test


X.to_numpy()
y.to_numpy()
testX.to_numpy()


# feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
testX = scaler.transform(testX)


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
model = RandomForestClassifier(n_estimators=115,max_depth=10)
model.fit(X,y)
y_pred = model.predict(testX)
y_pred = y_pred.astype(int)
df = pd.read_csv("out.csv")
df["Label"] = y_pred
df.to_csv("out.csv")

