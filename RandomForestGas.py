#!/usr/bin/python3

# RandomForestGas.py
# This code was written on December 2018 for a class project (Machine Learning Course)
# Sima Esfandiarpour Borujeni

# "This code uses Random Forest method from scikit-learn to predict the concentration level...
# ...of 6 gas types based on the measurementsfrom 16 sensors. The dataset is split to test...
# ...and train set with 70% and 30% of the data points respectively. The process is repeated...
# ... with 1,30,200,800, 2000 estimator to check the accuracy of the prediction..."

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#### Importing data###
batch1=pd.read_csv('batch1.dat', delimiter="\ +[0-9]+:{1}", skipinitialspace=True, header=None, engine='python')
batch2=pd.read_csv('batch2.dat', delimiter="\ +[0-9]+:{1}", skipinitialspace=True, header=None, engine='python')
batch3=pd.read_csv('batch3.dat', delimiter="\ +[0-9]+:{1}", skipinitialspace=True, header=None, engine='python')
batch4=pd.read_csv('batch4.dat', delimiter="\ +[0-9]+:{1}", skipinitialspace=True, header=None, engine='python')
batch5=pd.read_csv('batch5.dat', delimiter="\ +[0-9]+:{1}", skipinitialspace=True, header=None, engine='python')
batch6=pd.read_csv('batch6.dat', delimiter="\ +[0-9]+:{1}", skipinitialspace=True, header=None, engine='python')
batch7=pd.read_csv('batch7.dat', delimiter="\ +[0-9]+:{1}", skipinitialspace=True, header=None, engine='python')
batch8=pd.read_csv('batch8.dat', delimiter="\ +[0-9]+:{1}", skipinitialspace=True, header=None, engine='python')
batch9=pd.read_csv('batch9.dat', delimiter="\ +[0-9]+:{1}", skipinitialspace=True, header=None, engine='python')
batch10=pd.read_csv('batch10.dat', delimiter="\ +[0-9]+:{1}", skipinitialspace=True, header=None, engine='python')

gasdata= pd.concat([batch1, batch2, batch3,batch4,batch5,batch6,batch7,batch8,batch9,batch10], ignore_index=True)
X = gasdata.iloc[:, 1:]
Y = gasdata.iloc[:, 0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=33)

###  Training the random forest model

forest1 = RandomForestClassifier(n_estimators=1)
forest1 = forest1.fit(X_train, Y_train)
Y_predict1 = forest1.predict(X_test)

# representing the results and calculating the error
error = 0
i = 0
for test in Y_test:
    if test != Y_predict1[i]:
        error += 1
    i+=1
error = float(error)
NumberOfClassifications = float(len(Y_predict1))

classificationError = (error/NumberOfClassifications) * 100

print("for the random forest using n_estimators=1: " , Y_predict1)
print("Number of mispredicted classifications: %d" % error)
print("number of predicted classes: %d" % NumberOfClassifications)
print("error of classification: %0.3f" % classificationError)

#############################################################################################
#Varying the number of estimators to 10, 100, 1000 respectively

forest30 = RandomForestClassifier(n_estimators=30)
forest30 = forest30.fit(X_train, Y_train)
Y_predict30 = forest30.predict(X_test)


error = 0
i = 0
for test in Y_test:
    if test != Y_predict30[i]:
        error += 1
    i+=1
error = float(error)
NumberOfClassifications = float(len(Y_predict30))

classificationError = (error/NumberOfClassifications) * 100

print("for the random forest using n_estimators=30: " , Y_predict30)
print("Number of mispredicted classifications: %d" % error)
print("number of predicted classes: %d" % NumberOfClassifications)
print("error of classification: %0.3f" % classificationError)


##################################################################################################

forest200 = RandomForestClassifier(n_estimators=200)
forest200 = forest200.fit(X_train, Y_train)
Y_predict200 = forest200.predict(X_test)

### Representing the results and calculating the error ###

error = 0
i = 0
for test in Y_test:
    if test != Y_predict200[i]:
        error += 1
    i+=1
error = float(error)
NumberOfClassifications = float(len(Y_predict200))

classificationError = (error/NumberOfClassifications) * 100

print("for the random forest using n_estimators=200: " , Y_predict200)
print("Number of mispredicted classifications: %d" % error)
print("number of predicted classes: %d" % NumberOfClassifications)
print("error of classification: %0.3f" % classificationError)

#########################################################################################
forest800 = RandomForestClassifier(n_estimators=800)
forest800 = forest800.fit(X_train, Y_train)
Y_predict800 = forest800.predict(X_test)

error = 0
i = 0
for test in Y_test:
    if test != Y_predict800[i]:
        error += 1
    i+=1
error = float(error)
NumberOfClassifications = float(len(Y_predict800))

classificationError = (error/NumberOfClassifications) * 100

print("for the random forest using n_estimators=800: " , Y_predict800)
print("Number of mispredicted classifications: %d" % error)
print("number of predicted classes: %d" % NumberOfClassifications)
print("error of classification: %0.3f" % classificationError)


########################################################################################
forest2000 = RandomForestClassifier(n_estimators=2000)
forest2000 = forest800.fit(X_train, Y_train)
Y_predict2000 = forest2000.predict(X_test)


error = 0
i = 0
for test in Y_test:
    if test != Y_predict2000[i]:
        error += 1
    i+=1
error = float(error)
NumberOfClassifications = float(len(Y_predict2000))

classificationError = (error/NumberOfClassifications) * 100

print("for the random forest using n_estimators=2000: " , Y_predict2000)
print("Number of mispredicted classifications: %d" % error)
print("number of predicted classes: %d" % NumberOfClassifications)
print("error of classification: %0.3f" % classificationError)



