import re
import sys
import io
import csv
import nltk
import preprocessing
import nltk
import pandas as pd

def status_processing(corpus):
    #myCorpus = preprocessing.NLTKPreprocessor(corpus)
    myCorpus = preprocessing.PreProcessing(corpus)
    #myCorpus.text = str(corpus)

    #fuck this stuff
    #print ("Doing the Initial Process...")
    #myCorpus.initial_processing()

    #print ("Starting Lexical Diversity...")
    myCorpus.lexical_diversity()

    #print ("Removing Stopwords...")
    myCorpus.stopwords()
 
    #print ("Lemmatization...")
    myCorpus.lemmatization()

    #print ("POS tagging....")
    myCorpus.pos_tag()

    return (myCorpus.stopwords()) ##only works when returning myCorpus.stopwords? unsure why 

    #return (myCorpus)
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
    print ("End of the Pre-Processing Process")
    print (word_final)
    
with open('output.csv', 'w') as csvFile:
    fieldnames  = ['Words', 'Token']
    writer = csv.DictWriter(csvFile, fieldnames = fieldnames)
    writer.writeheader()

    i = 0
    for i in range(len(word_final)):
        writer.writerow({'Words': word_final[i][0], 'Token': word_final[i][1]})

    #file = open("output.csv","w")
    #file.write(word_final)
    #file.close()





