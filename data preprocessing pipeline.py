import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:/Users/Tharuni/Desktop/NIT/Aug month/13th-stats work/Data (1).csv")
dataset.describe()

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values #separating x and y variables

from sklearn.impute import SimpleImputer #transformers to fill numerical missing value
imputer = SimpleImputer(strategy='mean')
imputer = imputer.fit(x[:,1:3]) # fit only on 2nd and 3rd columns

x[:,1:3] = imputer.transform(x[:,1:3]) #replacement of missing values will happen 


from sklearn.preprocessing import LabelEncoder # to convert categorical to numerical data and creating dummy var

labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# model training and testing ratios
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.8,train_size=0.2)
#x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.2)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.2,random_state=0)
#when u add random_state=0 in train test split 
