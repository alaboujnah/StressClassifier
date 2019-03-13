# coding: utf-8

import pandas as pd
import sklearn.utils
from datashape import  np
from sklearn import metrics, svm
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

N = pd.read_csv('C:\Python27\data.csv', sep=',',header=0,error_bad_lines=False)

sklearn.utils.shuffle(N)
#print(N)
Y=['bpm', 'rmssd', 'bsv', 'sdnn']
Xlabel=N["state"].values
Yfeture=N[list(Y)].values

#print("permutation")
#print(P)


#training, test = P[X,:0.33], P[Y,:0.33]
train_P, test_P, train_labels, test_labels = train_test_split(Yfeture,Xlabel, test_size=0.33)

# we create an instance of SVM and fit out data #c is SVM regularization parameter


params = {'kernel': 'linear'}
#initialize the classifier
scv = SVC(**params)
#trainning the classifier
scv.fit(train_P, train_labels)
#svc = svm.SVC(kernel='linear', C=0.1).fit(train_P, train_labels)


#preductions

predicted = scv.predict(test_P)

#confusion
#C = confusion_matrix(test_labels.predict(X))
C=confusion_matrix(test_labels, predicted)
print("confusion",C)

accuracy = scv.score(train_P, train_labels)
print("accurancy ",accuracy*100)
