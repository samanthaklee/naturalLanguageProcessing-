##analysis functions not written by me 

import io
import sys
import matplotlib as plot #visuals
#from wordcloud import WordCloud, STOPWORDS #create a dank wordcloud
import matplotlib.pyplot as plt
import string

import sklearn as skl
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2

from collections import defaultdict, Counter
import operator
from pprint import pprint

import nltk as nltk #nlp module
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import treebank
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords

import random
from random import sample

from collections import Counter
from subprocess import check_output

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score
cv = CountVectorizer()
from sklearn.metrics.pairwise import cosine_similarity
stop = set(stopwords.words("english"))
import warnings
warnings.filterwarnings('ignore')


def extract_and_train():
    train, test = train_test_split(newsPred, test_size = 0.20, random_state = 42)
    # Create the training test omitting the diagnosis

    training_set = train.ix[:, train.columns != 'Accuracy']
    # Next we create the class set 
    class_set = train.ix[:, train.columns == 'Accuracy']

    # Next we create the test set doing the same process as the training set
    test_set = test.ix[:, test.columns != 'Accuracy']
    test_class_set = test.ix[:, test.columns == 'Accuracy']

    # Call a Naive Bayes and Linear SVM algorithm on the data
    naive_bayes(training_set, class_set, test_set, test_class_set)
    linear_svm(training_set, class_set, test_set, test_class_set)

def naive_bayes(training_set, class_set, test_set, test_class_set):
    # Building a Pipeline; this does all of the work in extract_and_train() at once  
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])
 
    text_clf = text_clf.fit(training_set, class_set)
    
    # Evaluate performance on test set
    predicted = text_clf.predict(test_set)
    print("The accuracy of a Naive Bayes algorithm is: ") 
    print(np.mean(predicted == test_class_set))
    print("Number of mislabeled points out of a total %d points for the Naive Bayes algorithm : %d"
          % (test_set.shape[0],(test_class_set != predicted).sum()))
    
    # Tune parameters and predict unlabeled tweets
    parameter_tuning(text_clf, training_set, class_set)
    predict_unlabeled_tweets(text_clf, predicted_data_NB)

def linear_svm(training_set, class_set, test_set, test_class_set):
    """ Let's try a Linear Support Vector Machine (SVM) """

    text_clf_two = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(
                                 loss='hinge',
                                 penalty='l2',
                                 alpha=1e-3,
                                 n_iter=5,
                                 random_state=42)),
    ])
    text_clf_two = text_clf_two.fit(training_set, class_set)
    predicted_two = text_clf_two.predict(test_set)
    print("The accuracy of a Linear SVM is: ")
    print(np.mean(predicted_two == test_class_set))
    print("Number of mislabeled points out of a total %d points for the Linear SVM algorithm: %d"
          % (test_set.shape[0],(test_class_set != predicted_two).sum()))
    
    # Tune parameters and predict unlabeled tweets
    parameter_tuning(text_clf_two, training_set, class_set)
    predict_unlabeled_tweets(text_clf_two, predicted_data_LSVM)

def parameter_tuning(text_clf, training_set, class_set):
    """ 
    Classifiers can have many different parameters that can make the                                                                                                             
    algorithm more accurate (MultinomialNB() has a smoothing                                                                                                                            parameter, SGDClassifier has a penalty parameter, etc.). Here we                                                                                                                    will run an exhaustive list of the best possible parameter values 
    """
    
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
                  }
    
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    
    gs_clf = gs_clf.fit(training_set, class_set)
    
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
        
        
        print(score)

def predict_unlabeled_tweets(classifier, output):
    # Make predictions
    unlabeled_tweets = pd.read_csv(unlabeled_data, names = unlabeled_columns)
    unlabeled_words = np.array(unlabeled_tweets["tweet"])
    predictions = classifier.predict(unlabeled_words)
    print(predictions)
    
    # Create new file for predictions
    # And utilize csv module to iterate through csv
    predicted_tweets = csv.writer(open(output, "wb+"))
    unlabeled_tweets = csv.reader(open(unlabeled_data, "rb+"))
    
    # Iterate through csv and get president and tweet
    # Add prediction to end
    # Also recieved from Github:
    # http://stackoverflow.com/questions/23682236/add-a-new-column-to-an-existing-csv-file
    index = 0
    for row in unlabeled_tweets:
        ('Json', 'Accuracy', 'Summary', 'Genre', 'KeywordName', 'Occupation', 'Location', 'PoliticalParty', 
                    '1', '2', '3', '4', '5', 'Source', 'Url' = row
        predicted_tweets.writerow([president] + [tweet] + [predictions[index]])
        index += 1

def compare_predictions():
    # Compare the predictions of both algorithms
    names = ["Accuracy", "Title", "prediction"]
    naive_bayes = pd.read_csv(factCheck.csv, names = names) 
    linear_svm = pd.read_csv(factCheck.csv, names = names) 

    naive_bayes_pred = np.array(naive_bayes["prediction"])
    linear_svm_pred = np.array(linear_svm["prediction"])
    
    print("The precent similarity between a Multinomial Naive Bayes Algorithm and a Linear SVM algorithm with a SGD Classifier is: ")
    print(np.mean(naive_bayes_pred == linear_svm_pred))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'tweets':
            get_tweets()
        elif sys.argv[1] == 'parse':
            parse_csv()
        elif sys.argv[1] == 'train':
            extract_and_train()
        elif sys.argv[1] == 'compare':
            compare_predictions()
    