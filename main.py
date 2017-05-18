import re
import sys
import io
import preprocessing
import pandas as pd

def status_processing(corpus):
    #myCorpus = preprocessing.NLTKPreprocessor(corpus)
    myCorpus = preprocessing.PreProcessing(corpus)
    #myCorpus.text = str(corpus)

    print ("Doing the Initial Process...")
    #myCorpus.initial_processing()

    print ("Starting Lexical Diversity...")
    myCorpus.lexical_diversity()

    print ("Removing Stopwords...")
    myCorpus.stopwords()
 
    print ("Lemmatization...")
    myCorpus.lemmatization()

    #print ("Correcting the words...")
    #myCorpus.spell_correct()
    #print ("Done")
    #print ("----------------------------")

    #print ("Untokenizing...")
    #word_final = myCorpus.untokenizing()

    #return word_final


if __name__ == '__main__':
    #dtype_dic = {'Summary': str}
    newsPred_columns = ['Json', 'Accuracy', 'Summary', 'Genre', 'KeywordName', 'Occupation', 'Location', 'PoliticalParty', 
                    '1', '2', '3', '4', '5', 'Source', 'Url']
    txt_corpus = pd.read_csv('factCheck.csv', 
        names = newsPred_columns)
  
    #txt_corpus['Summary'] = txt_corpus['Summary'].astype(str)
    summary = str(txt_corpus['Summary'])
    #print (txt_corpus.dtypes)
    word_final = status_processing(summary)
    print ("End of the Pre-Processing Process ")
    print (word_final)