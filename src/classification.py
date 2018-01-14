#function not written by me but i wanted to see if it works 


import NLPanalysis
from sklearn.cross_validation import train_test_split
import pandas as pd
import re
import sys 
import io

newsPred = pd.read_csv('factCheck.csv')

index = 0
correct = 0

#train, test = train_test_split(newsPred, test_size = 0.20, random_state = 42)
    # Create the training test omitting the diagnosis

#training_set = train.ix[:, train.columns != 'Accuracy']
    # Next we create the class set 
#class_set = train.ix[:, train.columns == 'Accuracy']

    # Next we create the test set doing the same process as the training set
#test_set = test.ix[:, test.columns != 'Accuracy']
#test_class_set = test.ix[:, test.columns == 'Accuracy']


for accuracy in newsPred:
        #(president, tweet, predicted) = row
        print(accuracy)
        sentiment = input("Enter your sentiment: ")
        if str(training_set) == str(test_set):
        	correct += 1
        index += 1

print("The percent similarity our guesses were to the algorithm were: %d" % ((float(correct / index)) * 100))