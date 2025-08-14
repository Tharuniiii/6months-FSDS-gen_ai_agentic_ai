import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Tharuni\Desktop\NIT\Aug month\14th-regression\Salary_Data.csv")

#dividing the dataset
x = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#model building we need x_train,y_train
from sklearn.linear_model import LinearRegression #importing algorithm
regressor = LinearRegression() #regressor is model
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)#predicted table will come

#plotting the graph for model predictions
plt.scatter(x_test,y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#prediction of future data
m = regressor.coef_
c = regressor.intercept_
(m*12)+c
(m*20)+c



