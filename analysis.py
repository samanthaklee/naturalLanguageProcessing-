import preprocessing
import main
import re
import sys 
import pandas as pd
import numpy as np
import csv
import random
import time
from time import strftime
#from pandas.DataFrame import query
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import nltk
from nltk import corpus
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
#nltk.download('punkt')
#nltk.download('stopwords')
 
#from sklearn import datasets
#from sklearn.naive_bayes import GaussianNB
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.pipeline import Pipeline
#from sklearn.linear_model import SGDClassifier
#from sklearn import metrics
#from sklearn.grid_search import GridSearchCV


def parse_csv():
    # Here we will parse a CSV file with the data on Row ID, Tweet ID,
    # Timestamp, President, Tweet

    training_file = csv.writer(open(training_data, "wb+"))
    testing_file = csv.writer(open(testing_data, "wb+"))
    unlabeled_file = csv.writer(open(unlabeled_data, "wb+"))
        
    # Now to randomize the data; this is how
    # Gotten from Github: 
    # (http://stackoverflow.com/questions/4618298/randomly-mix-lines-of-3-million-line-file)
    with open(file, 'rb') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(randomized_file, 'wb+') as target:
        for _, line in data:
            target.write( line )
    
    prepped_tweet_file = csv.reader(open(randomized_file, "rb"))
    index = 0

    # Now we will iterate through the randomized file and extract data
    # We need to get rid of the decimal points in the seconds columns
    # And then split up the data (2/3 train and 1/3 test)
    # And obtain frequencies as well
    
    for row in prepped_tweet_file:
        (row_id, tweet_id, timestamp, president, tweet, label) = row
        raw_timestamp = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
        ratio = 3
        
        # Take care of unlabeled data
        if label == "1":
            tokenize_row_write(unlabeled_file, row_id, tweet_id, raw_timestamp.tm_mon, raw_timestamp.tm_wday, raw_timestamp.tm_hour, president, tweet, "")
            continue
 
   
        # Get frequencies of sentiment per day of week
        # Index values of array used for day of week
        if label == "positive":
            pos_freq[raw_timestamp.tm_wday] += 1
        if label == "negative":
            neg_freq[raw_timestamp.tm_wday] += 1
        else:
            neu_freq[raw_timestamp.tm_wday] += 1
            
        # Now do this for each candidate
        if president == "HillaryClinton":
            if label == "positive":
                pos_freq_clinton[raw_timestamp.tm_wday] += 1
            if label == "negative":
                neg_freq_clinton[raw_timestamp.tm_wday] += 1
            else:
                neu_freq_clinton[raw_timestamp.tm_wday] += 1

        if president == "realDonaldTrump":
            if label == "positive":
                pos_freq_trump[raw_timestamp.tm_wday] += 1
            if label == "negative":
                neg_freq_trump[raw_timestamp.tm_wday] += 1
            else:
                neu_freq_trump[raw_timestamp.tm_wday] += 1

   
        if index % ratio == 0:
            tokenize_row_write(testing_file, row_id, tweet_id, raw_timestamp.tm_mon, raw_timestamp.tm_wday, raw_timestamp.tm_hour, president, tweet, label)
        else:
            tokenize_row_write(training_file, row_id, tweet_id, raw_timestamp.tm_mon, raw_timestamp.tm_wday, raw_timestamp.tm_hour, president, tweet, label)
            