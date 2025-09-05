import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\Tharuni\Desktop\NIT\sept month\3rd,4th_confusion matrix practicle\data\logit classification.csv")

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.20,random_state=0)

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# feature scaling normalizer
#from sklearn.preprocessing import Normalizer
#sc = Normalizer()
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)

# model building
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

# model accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#accuracy score
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)
# classification report 
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

# bias and varience to find overfit or underfit
bias= classifier.score(x_train, y_train)
print(bias)

variance = classifier.score(x_test,y_test)
print(variance)
# this was best fit model bias and variance is model accuracy is similar

# Future prediction
dataset1 = pd.read_csv(r"C:\Users\Tharuni\Desktop\NIT\sept month\3rd,4th_confusion matrix practicle\data\final1.csv")
d2 = dataset1.copy()
dataset1 = dataset1.iloc[:,[3,4]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()
d2 ['y_pred1'] = classifier.predict(M)# passing the data to future predictions

d2.to_csv("final1.csv")
import os
os.getcwd()







