import nltk
import io
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
#from bs4 import BeautifulSoup
#import spellcorrect


class PreProcessing():
    def __init__(self, text):
        self.text = text
        #soup = BeautifulSoup(text, "html.parser")
        #Todo Se quiser salvar os links mudar aqui
        #self.text = re.sub(r'(http://|https://|www.)[^"\' ]+', " ", soup.get_text())
        self.tokens = self.tokenizing()
        #return (self.tokens)

    def lexical_diversity(self):
        """
        lexical density provides a measure of the proportion of lexical items 
        (i.e. nouns, verbs, adjectives and some adverbs) in the text.
        """
        word_count = len(self.text)
        vocab_size = len(set(self.text))
        return vocab_size / word_count

    def tokenizing(self, use_default_tokenizer=True):
        if use_default_tokenizer:
            return nltk.tokenize.word_tokenize(self.text)
       # words = word_tokenize(self.text)
        stok = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        return stok.tokenize(self.text)

    def stopwords(self):
        stopwords = set(nltk.corpus.stopwords.words('portuguese'))
        stopwords.update([
            'foda', 'caralho', 'porra',
            'puta', 'merda', 'cu',
            'foder', 'viado', 'cacete'])

        self.tokens = [word for word in self.tokens if word not in stopwords]
        return (self.tokens)

    def stemming(self):
        """
        stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root 
        form generally a written word form.
        """

        snowball = SnowballStemmer('portuguese')
        self.tokens = [snowball.stem(word) for word in self.tokens]
        return (self.tokens)

    def lemmatization(self):

        """
        Lemmatisation (or lemmatization) in linguistics, is the process of grouping together the different inflected 
        forms of a word so they can be analysed as a single item.
        """
        lemmatizer = WordNetLemmatizer()  #'portuguese'
        self.tokens = [lemmatizer.lemmatize(word, pos='v') for word in self.tokens]
        return (self.tokens)

    def pos_tag(self):
        """
        Use NLTK's currently recommended part of speech tagger to
        tag the given list of tokens.
        """

        self.tokens = nltk.pos_tag(self.tokens)
        return (self.tokens)

    def part_of_speech_tagging(self):
        raise NotImplementedError

    def untokenize(self, words):
        """
        Untokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `untokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.
        """
        text = ' '.join(words)
        step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
             "can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        return step6.strip()

    def untokenizing(self):
        return ' '.join(self.tokens)

  #  def spell_correct(self):
   #     self.tokens = [spellcorrect.correct(word) for word in self.tokens]