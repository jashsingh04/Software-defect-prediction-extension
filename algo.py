import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import sys
import csv
from selector import PATH
import seaborn as sns  # statistical data visualization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

from selector import ALGORITHM

from importsall import  df

from BAT import ttsbat

from SparrowSearch import ttsssa

from SquirrelSearch import ttsqsa

if ALGORITHM == 'Sparrow Search':
    X_train, X_test, y_train, y_test = ttsssa[0], ttsssa[1], ttsssa[2], ttsssa[3]

elif ALGORITHM == 'BAT':
    X_train, X_test, y_train, y_test = ttsbat[0], ttsbat[1], ttsbat[2], ttsbat[3]

else :
     X_train, X_test, y_train, y_test = ttsqsa[0], ttsqsa[1], ttsqsa[2], ttsqsa[3]

ACCURACY, PRECISION, RECALL, AUC, F1 = [], [], [], [], []


clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

#print("Random Forest Classifier\n")
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
ACCURACY.append(accuracy)
#print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred)
PRECISION.append(precision)
#print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
RECALL.append(recall)

#print("Recall:", recall)

# Calculate f1 score
f1 = f1_score(y_test, y_pred)
F1.append(f1)

#print("F1 Score:", f1)

auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
AUC.append(auc)
#print("AUC:", auc)






clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#print("Gaussian Naive Bayes\n")
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
ACCURACY.append(accuracy)
#print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred)
PRECISION.append(precision)
#print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
RECALL.append(recall)

#print("Recall:", recall)

# Calculate f1 score
f1 = f1_score(y_test, y_pred)
F1.append(f1)
#print("F1 Score:", f1)

# Calculate AUC score
auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
AUC.append(auc)

#print("AUC:", auc)








#print("Ada Boost Classifier\n")

clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
ACCURACY.append(accuracy)
#print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred)
PRECISION.append(precision)
#print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
RECALL.append(recall)

#print("Recall:", recall)

# Calculate f1 score
f1 = f1_score(y_test, y_pred)
F1.append(f1)
#print("F1 Score:", f1)

# Calculate AUC score
auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
AUC.append(auc)

#print("AUC:", auc)








clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

#print("Multi Layer Perceptron\n")
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
PRECISION.append(precision)
ACCURACY.append(accuracy)
#print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred)
#print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
RECALL.append(recall)

#print("Recall:", recall)

# Calculate f1 score
f1 = f1_score(y_test, y_pred)
F1.append(f1)
#print("F1 Score:", f1)

# Calculate AUC score
auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
AUC.append(auc)

#print("AUC:", auc)








clf = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

#print("KNeighbors Classifier\n")
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
ACCURACY.append(accuracy)
#print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred)
PRECISION.append(precision)
#print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
RECALL.append(recall)
#print("Recall:", recall)

# Calculate f1 score
f1 = f1_score(y_test, y_pred)
F1.append(f1)
#print("F1 Score:", f1)

# Calculate AUC score
auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
AUC.append(auc)

#print("AUC:", auc)
















#CNN

sns.set_theme(style="whitegrid")

df.describe()
df.tail()
X = df.iloc[:, 1:-1].values

y = df.iloc[:,-1].values

xaxis = df['defects'].unique()

yaxis = df['defects'].value_counts()

sns.barplot(x=xaxis,y=yaxis)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## Naive bayers
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
#print("Number of mislabeled points out of a total %d points : %d"      % (X_test.shape[0], (y_test != y_pred).sum()))
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
##print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
## SVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(),SVC(kernel='linear',C=1))
clf.fit(X_train,y_train)
y_pred_svm = clf.predict(X_test)

#print("Number of mislabeled points out of a total %d points : %d"
# % (X_test.shape[0], (y_test != y_pred_svm).sum()))
from sklearn.metrics import accuracy_score
accuracy_score(y_pred_svm,y_test)*100
# CNN
## setting up of CNN modules

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D




input_shape = X_train.shape[1]


model_m = Sequential()
model_m.add(Reshape((input_shape, 1), input_shape=(input_shape,)))
model_m.add(Conv1D(100, input_shape, activation='relu'))

# model_m.add(Conv1D(100, 11, activation='relu'))
# model_m.add(MaxPooling1D(1))
# model_m.add(Conv1D(160, 3, activation='relu'))
# model_m.add(Conv1D(160, 3, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(1, activation='tanh'))
##print(model_m.summary())
model_m.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])

y_train.astype('float64')
EPOCHS = 100
BATCH_SIZE = 64
history = model_m.fit(X_train,
                      y_train.astype('float64'),
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=1)


#print(model_m.summary())

accuracies = history.history['accuracy']

# Compute the average accuracy across all epochs
avg_accuracy = sum(accuracies) / len(accuracies)

#print(f"Average accuracy across {len(accuracies)} epochs: {avg_accuracy}")

# ACCURACY.append(avg_accuracy)
# F1.append('NaN')
# RECALL.append('NaN')
# PRECISION.append('NaN')
# AUC.append('NaN')





