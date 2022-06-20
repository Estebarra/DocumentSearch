import re
import spacy
from gensim import corpora
# importing stop words
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords

# loading tokenizer for english language
spacy_nlp_eng = spacy.load('en_core_web_sm')
spacy_nlp_esp = spacy.load('es_core_news_md')

# create list of punctuations and stopwords
english_stopwords = spacy.lang.en.stop_words.STOP_WORDS
spanish_stopwords = set(stopwords.words('spanish'))

# function for data cleaning and preprocessing
def tokenizer(sentence, lang): 
    # remove distracting single quotes
    sentence = re.sub('\'','',sentence)
    # remove digits and words containing digits
    sentence = re.sub('\w*\d\w*','',sentence)
    # replace extra spaces with single space
    sentence = re.sub(' +',' ',sentence)
    # remove unwanted lines starting from special characters
    sentence = re.sub(r'\n: \'\'.*','',sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)    
    # remove non-breaking new line characters
    sentence = re.sub(r'\n',' ',sentence)    
    # remove punctuations
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    
    # creating token object
    if lang == "eng":
        tokens = spacy_nlp_eng(sentence)
        # lower, strip and lemmatize
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]    
        # remove stopwords, and exclude words less than 2 characters
        tokens = [word for word in tokens if word not in english_stopwords and len(word) > 2]
    elif lang == "esp":
        tokens = spacy_nlp_esp(sentence)
        # lower, strip and lemmatize
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]    
        # remove stopwords, and exclude words less than 2 characters
        tokens = [word for word in tokens if word not in spanish_stopwords and len(word) > 2]
    
    return tokens


# function to create dictionary, corpus with search words and the words frequency
def generate_corpus(tokens, save_dict = False):
    # create dictionary
    dictionary = corpora.Dictionary()
    # corpus creation 
    corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tokens]
    # calcalating words frequency
    word_frequency = [[(dictionary[id], count) for id, count in doc] for doc in corpus]

    # saving dictionary and corpus
    if save_dict == True:
        dictionary.save('dictionary.dict')  
        corpora.MmCorpus.serialize('corpus.mm', corpus)

    return dictionary, corpus, word_frequency
