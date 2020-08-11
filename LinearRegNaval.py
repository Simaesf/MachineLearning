#linear regression Sima E. Borujeni 

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

naval=pd.read_csv('data.txt', delimiter='   ' , header=None) 
print(naval)  

X = naval.iloc[:, :16]
#print("X_train:")
#print(X_train)
Y = naval.iloc[:, -2]
#print("Y_train:")
print(Y) 
Z = naval.iloc[:, -1]
print(Z)

X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z, test_size=0.3, random_state=33)
print(Z_test)

reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)
#print(Y_pred)


reg = linear_model.LinearRegression()
reg.fit(X_train, Z_train)
Z_pred = reg.predict(X_test)
print (Z_pred)

print("Mean squared error: %.8f" % mean_squared_error(Y_test, Y_pred))                                      
print('Coefficients: \n', reg.coef_)


print("Mean squared error: %.8f" % mean_squared_error(Z_test, Z_pred))                                      
print('Coefficients: \n', reg.coef_)
