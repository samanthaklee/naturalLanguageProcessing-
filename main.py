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
    print ("Done.")
    print ("----------------------------")

    print ("StartingLexical Diversity...")
    myCorpus.lexical_diversity()
    print ("Done")
    print ("----------------------------")

    print ("Removing Stopwords...")
    myCorpus.stopwords()
    print ("Done")
    print ("----------------------------")

    print ("Lemmatization...")
    myCorpus.lemmatization()
    print ("Feito")
    print ("----------------------------")

    #print ("Correcting the words...")
    #myCorpus.spell_correct()
    #print ("Done")
    #print ("----------------------------")

    print ("Untokenizing...")
    word_final = myCorpus.untokenizing()
    print ("Feito")
    print ("----------------------------")

    return word_final


if __name__ == '__main__':
    #dtype_dic = {'Summary': str}
    newsPred_columns = ['Json', 'Accuracy', 'Summary', 'Genre', 'KeywordName', 'Occupation', 'Location', 'PoliticalParty', 
                    '1', '2', '3', '4', '5', 'Source', 'Url']
    txt_corpus = pd.read_csv('factCheck.csv', 
        names = newsPred_columns)
        #dtype=dtype_dic
        #encoding='utf-8', sep=',',
        #header='infer', engine='c', chunksize=1)
    txt_corpus['Summary'] = txt_corpus['Summary'].astype(str)
    summary = txt_corpus['Summary']

    word_final = status_processing(summary)

   # print ("Saving in DB....")
   # try:
     #   db.myDB.insert(word_final, continue_on_error=True)
    #except pymongo.errors.DuplicateKeyError:
      #  pass

    print ("End of the Pre-Processing Process ")