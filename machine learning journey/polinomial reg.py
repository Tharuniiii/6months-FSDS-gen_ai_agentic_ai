import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Tharuni\Desktop\NIT\Aug month\24th-non linear\employee salary prediction\emp_sal.csv")

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#simple linear with degree 1
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()#model
lin_reg.fit(x,y)

#linear regression visualisation
plt.scatter(x,y, color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.title('linear regression graph')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#prediction for simple linear regression
lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)#hyper parameter tuning

#polynomial model
from sklearn.preprocessing import PolynomialFeatures# by default this algo contains 2nd degree
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

#polynomial reg visualisation 
plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)),color = 'black')
plt.title('truth or bluff polynomial regression')
plt.xlabel('postition level')
plt.ylabel('salary')
plt.show()

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)
