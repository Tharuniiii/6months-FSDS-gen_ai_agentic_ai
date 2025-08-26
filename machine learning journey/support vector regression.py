import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Tharuni\Desktop\NIT\Aug month\24th-non linear\employee salary prediction\emp_sal.csv")

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#SVR model(Regression model)
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'poly',degree=5,gamma='auto',C =10.4)#hyperparameter tuning for kernal sigmoid
svr_regressor.fit(x,y)

svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)

# KNN Model
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=7,weights='uniform')
knn_reg_model.fit(x,y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)
