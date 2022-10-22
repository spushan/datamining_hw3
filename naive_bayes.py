#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#reading the training data
train_df = pd.read_csv('weather_training.csv')
train_df.drop(['Formatted Date'], axis=1, inplace=True)
training_arr = np.array(train_df.values).astype('f')
x_train = training_arr[:,:-1]
y_train = training_arr[:,-1]

#update the training class values according to the discretization (11 values only)
y_train = np.digitize(y_train,classes)

#reading the test data
test_df = pd.read_csv('weather_test.csv')
test_df.drop(['Formatted Date'], axis=1, inplace=True)
test_arr = np.array(test_df.values).astype('f')
x_test = test_arr[:,:-1]
y_test = test_arr[:,-1]
#update the test class values according to the discretization (11 values only)
y_test = np.digitize(y_test,classes)

#fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(x_train, y_train)

#make the naive_bayes prediction for each test sample and start computing its accuracy
#the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
#to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
correct=0
for x, y in zip(x_test, y_test):
    pred = clf.predict([x])
    diff = 100*(abs(pred[0] - y)/y)
    if diff <= 15:
        correct += 1
acc = correct/len(y_test)

#print the naive_bayes accuracyy
#--> add your Python code here
print("naive_bayes accuracy: " + str(acc))



