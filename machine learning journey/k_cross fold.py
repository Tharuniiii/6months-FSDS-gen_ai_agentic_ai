import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Tharuni\Desktop\NIT\sept month\3rd,4th_confusion matrix practicle\data\Social_Network_Ads.csv")

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:,-1].values
print(x)
print(y)

#encoding categorical data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
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

#k fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X= x_train,y = y_train,cv=5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

# Roc and Auc curve display
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

#roc curve
fpr, tpr, thresholds = roc_curve(y_test,y_pred)
roc_curve=auc(fpr,tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_curve:.2f})")
plt.plot([0,1], [0,1], color="red", linestyle="--", label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()




