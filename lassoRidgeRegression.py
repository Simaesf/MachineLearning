
# Project -> Lasso Ridge Regression

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

## Read data into vectors

inputData=pd.read_csv('data.txt', delim_whitespace = True, header=None)

featureSet = inputData.iloc[:, :-2]
cDecaySet = inputData.iloc[:, -2]
tDecaySet = inputData.iloc[:, -1]

xTrain, xTest, cTrain, cTest, tTrain, tTest = train_test_split(featureSet, cDecaySet, tDecaySet, test_size=0.3)
tXTrain, tXTest, tYTrain, tYTest = train_test_split(featureSet, tDecaySet, test_size=0.3)

## Train Ridge Regression models

cDecayModel = linear_model.Lasso(alpha=10.0)
cDecayModel.fit(xTrain, cTrain)

tDecayModel = linear_model.Lasso(alpha=10.0)
tDecayModel.fit(xTrain, tTrain)

## Predict and calculate error

cDecayPrediction = cDecayModel.predict(xTest)
cDecayError = mean_squared_error(cTest, cDecayPrediction)

tDecayPrediction = tDecayModel.predict(tXTest)
tDecayError = mean_squared_error(tTest, tDecayPrediction)

## Print results

print
print("----- Results -----")
print

print("\tCompressor Decay Model:" )
print("Error: %.8f" % cDecayError)
print("Coefficients:")
print(cDecayModel.coef_)

error = 0
for coef in cDecayModel.coef_:
    if coef == 0:
        error += 1
print("Number of zero coefficients: %d" % error)
print

print("\tTurbine Decay Model:")
print("Error: %.8f" % tDecayError)
print("Coefficients:")
print(tDecayModel.coef_)

error = 0
for coef in tDecayModel.coef_:
    if coef == 0:
        error +=  1
print("Number of zero coefficients: %d" % error)
print


