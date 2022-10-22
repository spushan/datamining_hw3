#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
train_df = pd.read_csv('weather_training.csv')
train_df.drop(['Formatted Date'], axis=1, inplace=True)
training_arr = np.array(train_df.values).astype('f')
#reading the test data
test_df = pd.read_csv('weather_test.csv')
test_df.drop(['Formatted Date'], axis=1, inplace=True)
test_arr = np.array(test_df.values).astype('f')
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
x_train = training_arr[:,:-1]
y_train = training_arr[:,-1]
x_test = test_arr[:,:-1]
y_test = test_arr[:,-1]
#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here
highest_acc=0
k_val = 0
p_val = 0
w_val = 0

for k in k_values:
    for v in p_values:
        for w in w_values:

            #fitting the knn to the data
            #--> add your Python code here

            #fitting the knn to the data
            clf = KNeighborsRegressor(n_neighbors=k, p=v, weights=w)
            clf = clf.fit(x_train, y_train)
             
            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            correct=0
            for x, y in zip(x_test, y_test):
                pred = clf.predict([x])
                diff = 100*(abs(pred - y)/y)
                if diff <=15:
                    correct += 1
            acc = correct/len(y_test)
               
                
                
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            #--> add your Python code here

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            #--> add your Python code here
            if acc > highest_acc:
                highest_acc = acc
                k_val = k
                p_val = v
                w_val = w
                
            print(f'Highest KNN accuracy so far: {highest_acc}, Parameters: k={k_val}, p={p_val}, w={w_val}')





