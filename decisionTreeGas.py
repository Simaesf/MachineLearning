# DTGas- Sima E. Borujeni

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

#importing data:
batch1=pd.read_csv('batch1.dat', delimiter="\ +[0-9]+:{1}", skipinitialspace=True, header=None, engine='python')
#print(batch1)
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
print(gasdata)
X = gasdata.iloc[:, 1:]
Y = gasdata.iloc[:, 0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=33)
#X_train
#Y_train
#X_test
#Y_test


#Training the decision tree model

Dtree = tree.DecisionTreeClassifier(min_samples_leaf=1)
Dtree = Dtree.fit(X_train, Y_train)
Y_predict = Dtree.predict(X_test)

#print("Y pred: " )
#print("len")
#print(len(Y_Predict))
#for i in range(len(Y_Predict)):
#    print(Y_Predict[i])
#print
#
#print(Y_test)

#print("Y_test: ")
#print("len")
#print(len(Y_test))
#for i in range(len(Y_test)):
#    print("here")
#    print("i:")
#    print(i)
#    print(Y_test[i])
#print
# representing the results and calculating error of classification

error = 0
i = 0
for test in Y_test:
    if test != Y_predict[i]:
        error += 1
    i+=1
error = float(error)
NumberOfClassifications = float(len(Y_predict))
classificationError = (error/NumberOfClassifications) * 100

print("Tree with min_samples_leaf=1: " , Y_predict)
print("Number of mispredicted classes: %d" % error)
print("number of predicted calsses: %d" % NumberOfClassifications)
print("classification error: %0.3f" % classificationError)


###############################################################################
#varying the minimum number of samples required at leaf node to 3,6,7 respectively
Dtree3 = tree.DecisionTreeClassifier(min_samples_leaf=3)
Dtree3 = Dtree3.fit(X_train, Y_train)
Y_predict3 = Dtree3.predict(X_test)


error = 0
i = 0
for test in Y_test:
    if test != Y_predict3[i]:
        error += 1
    i+=1
error = float(error)
NumberOfClassifications = float(len(Y_predict3))
classificationError = (error/NumberOfClassifications) * 100

print("Tree with min_samples_leaf=3: " , Y_predict3)
print("Number of mispredicted classes: %d" % error)
print("number of predicted calsses: %d" % NumberOfClassifications)
print("classification error: %0.3f" % classificationError)


###############################################################################

Dtree4 = tree.DecisionTreeClassifier(min_samples_leaf=4)
Dtree4 = Dtree4.fit(X_train, Y_train)
Y_predict4 = Dtree4.predict(X_test)

error = 0
i = 0
for test in Y_test:
    if test != Y_predict4[i]:
        error += 1
    i+=1
error = float(error)
NumberOfClassifications = float(len(Y_predict4))
classificationError = (error/NumberOfClassifications) * 100

print("Tree with min_samples_leaf=4: " , Y_predict4)
print("Number of mispredicted classes: %d" % error)
print("number of predicted calsses: %d" % NumberOfClassifications)
print("classification error: %0.3f" % classificationError)

###############################################################################

Dtree6 = tree.DecisionTreeClassifier(min_samples_leaf=6)
Dtree6 = Dtree6.fit(X_train, Y_train)
Y_predict6 = Dtree6.predict(X_test)

error = 0
i = 0
for test in Y_test:
    if test != Y_predict6[i]:
        error += 1
    i+=1
error = float(error)
NumberOfClassifications = float(len(Y_predict6))
classificationError = (error/NumberOfClassifications) * 100

print("Tree with min_samples_leaf=6: " , Y_predict6)
print("Number of mispredicted classes: %d" % error)
print("number of predicted calsses: %d" % NumberOfClassifications)
print("classification error: %0.3f" % classificationError)

###############################################################################

Dtree7 = tree.DecisionTreeClassifier(min_samples_leaf=7)
Dtree7 = Dtree7.fit(X_train, Y_train)
Y_predict7 = Dtree7.predict(X_test)


error = 0
i = 0
for test in Y_test:
    if test != Y_predict7[i]:
        error += 1
    i+=1
error = float(error)
NumberOfClassifications = float(len(Y_predict7))
classificationError = (error/NumberOfClassifications) * 100

print("Tree with min_samples_leaf=7: " , Y_predict7)
print("Number of mispredicted classes: %d" % error)
print("number of predicted calsses: %d" % NumberOfClassifications)
print("classification error: %0.3f" % classificationError)









