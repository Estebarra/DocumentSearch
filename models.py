import gensim
from gensim.similarities import MatrixSimilarity
import cleansing

# function that creates the tfidf and lsi models and the similarity matrix
def create_models(dictionary, corpus):
    # load tfidf model
    tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
    # load lsi model and extract corpus to be used in matrixsimilarity function
    lsi_model = gensim.models.LsiModel(tfidf_model[corpus], id2word=dictionary, num_topics=300)
    gensim.corpora.MmCorpus.serialize('lsi_model.mm',lsi_model[tfidf_model[corpus]])  
    lsi_corpus = gensim.corpora.MmCorpus('lsi_model.mm')
    # similarity indexes
    index = MatrixSimilarity(lsi_corpus, num_features = lsi_corpus.num_terms)

    return tfidf_model, lsi_model, index 

# Function that search the words in the documents
def search_similar_words(search_term, dictionary, tfidf_model, lsi_model, index, lang):
    # transform search terms to tokens and create the corpus
    query_bow = [dictionary.doc2bow(cleansing.tokenizer(search_term,lang))]
    # apply tfidf model to the corpus
    query_tfidf = tfidf_model[query_bow]
    # apply lsi model to the tfidf
    query_lsi = lsi_model[query_tfidf]
    # number of elements with the maximum relevance
    index.num_best = 5
    
    return index[query_lsi]