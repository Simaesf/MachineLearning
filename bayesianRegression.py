
# Project -> Bayesian Ridge Regression

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

## Read data into vectors

inputData=pd.read_csv('data.txt', delim_whitespace=True, header=None)

featureSet = inputData.iloc[:, :-2]
cDecaySet = inputData.iloc[:, -2]
tDecaySet = inputData.iloc[:, -1]

xTrain, xTest, cTrain, cTest, tTrain, tTest = train_test_split(featureSet, cDecaySet, tDecaySet, test_size=0.3)
tXTrain, tXTest, tYTrain, tYTest = train_test_split(featureSet, tDecaySet, test_size=0.3)

## Train Ridge Regression models

cDecayModel = linear_model.BayesianRidge(compute_score=True)
cDecayModel.fit(xTrain, cTrain)

tDecayModel = linear_model.BayesianRidge(compute_score=True)
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
print("Alpha: %d" % cDecayModel.alpha_)
print("Coefficients:")
print(cDecayModel.coef_)
print("Objective function scores:")
print(cDecayModel.scores_)
print

print("\tTurbine Decay Model:")
print("Error: %.8f" % tDecayError)
print("Alpha: %d" % tDecayModel.alpha_)
print("Coefficients:")
print(tDecayModel.coef_)
print("Objective function scores:")
print(tDecayModel.scores_)
print


