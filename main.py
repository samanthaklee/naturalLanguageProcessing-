import re
import sys
import io
import preprocessing
import pandas as pd

def status_processing(corpus):
    myCorpus = preprocessing.PreProcessing(#what the fuck goes here, corpus)
    myCorpus.text = str(corpus)

    print ("Doing the Initial Process...")
    myCorpus.initial_processing()
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

    print ("Correcting the words...")
    myCorpus.spell_correct()
    print ("Done")
    print ("----------------------------")

    print ("Untokenizing...")
    word_final = myCorpus.untokenizing()
    print ("Feito")
    print ("----------------------------")

    return word_final


if __name__ == '__main__':
    dtype_dic = {'status_id': str, 'status_message': str, 'status_published': str}
    txt_corpus = pd.read_csv(
        'factCheck.csv', dtype=dtype_dic,
        encoding='utf-8', sep=',',
        header='infer', engine='c', chunksize=2)

    word_final = status_processing(txt_corpus)

    print ("Saving in DB....")
    try:
        db.myDB.insert(word_final, continue_on_error=True)
    except pymongo.errors.DuplicateKeyError:
        pass

    print ("Insertion in the DB Completed. End of the Pre-Processing Process ")