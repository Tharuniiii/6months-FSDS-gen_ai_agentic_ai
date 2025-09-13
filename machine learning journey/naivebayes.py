import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Tharuni\Desktop\NIT\sept month\3rd,4th_confusion matrix practicle\model\Social_Network_Ads.csv")

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

#feature scaling
from sklearn.preprocessing import Normalizer
sc = Normalizer()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)

bias = classifier.score(x_train,y_train)
print(bias) # training set results

variance = classifier.score(x_test,y_test)
print(variance) #testing set results

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)