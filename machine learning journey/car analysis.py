import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score


dataset = pd.read_csv(r"C:\Users\Tharuni\Desktop\NIT\Aug month\21st_car analysis\car-mpg.csv")
dataset.head()

#we are dropining the categorical attributes and filling or replacing the missing values
#replace ? with nan.
dataset = dataset.drop(['car_name'],axis=1)
dataset['origin'] = dataset['origin'].replace({1: 'america', 2:'europe',3:'asia'})
dataset = pd.get_dummies(dataset,columns = ['origin'],dtype=int)
dataset = dataset.replace('?',np.nan)

dataset = dataset.apply(pd.to_numeric,errors ='ignore')
#fill missing values with median for numeric values
numeric_cols = dataset.select_dtypes(include=[np.number]).columns
dataset[numeric_cols] = dataset[numeric_cols].apply(lambda x: x.fillna(x.median()))
dataset.head()

#model building 

x = dataset.drop(['mpg'],axis = 1)#independent variable
y = dataset[['mpg']]#dependent variable

#scaling the data
x_s = preprocessing.scale(x)
x_s = pd.DataFrame(x_s, columns = x.columns)

y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s, columns = y.columns)

#split the data into training and testing
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=1)
x_train.shape

# Simple linear model
#fit the slr and find the coeff values
regression_model = LinearRegression()
regression_model.fit(x_train,y_train)

for idx, col_name in enumerate(x_train.columns):
    print('The coefficient for {} is {}'.format(col_name,regression_model.coef_[0][idx]))
intercept = regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))

# Regularized ridge regression
#reduces the high coeff to low coeff with alpha factor is lambda
ridge_model = Ridge(alpha = 0.4)
ridge_model.fit(x_train,y_train)
print('Ridge model coef: {}'.format(ridge_model.coef_))

# regularized lasso regression
lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(x_train,y_train)
print('Lasso model coef: {}'.format(lasso_model.coef_))

#Score comparision for accuracy of the model

#simple linear Regression
print(regression_model.score(x_train,y_train))
print(regression_model.score(x_test,y_test))

#ridge regularisation
print(ridge_model.score(x_train,y_train))
print(ridge_model.score(x_test,y_test))

#lasso regularisation
print(lasso_model.score(x_train,y_train))
print(lasso_model.score(x_test,y_test))
