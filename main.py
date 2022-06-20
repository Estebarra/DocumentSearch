# required before run: 
# ----------------------------------------
# pip install spacy
# pip install gensim
# pip install python-Levenshtein (not necessary but prevent an initial warning)
# python -m spacy download en_core_web_sm
# ----------------------------------------

import gensim
from gensim import corpora
import parsing
import cleansing
import models
import pandas as pd

from datetime import datetime
from os import listdir
from os.path import isfile,join

THRESHOLD = 0.01

lang = input('Which language would you like to work with (esp/eng)?\n')

path = "./files_{}/".format(lang)

files = [f for f in listdir(path)]

# reading pdfs
texts = []
for pdf_file in files:
    texts.append(parsing.extractPDFText(path + pdf_file))

# extract words from text
doc_tokenized = [cleansing.tokenizer(text, lang) for text in texts]

# Generate the dictionary and corpus
dictionary, corpus, _ = cleansing.generate_corpus(doc_tokenized, save_dict=True)

# Generate models and similarity indexes
tfidf_model, lsi_model, index = models.create_models(dictionary, corpus)

now = datetime.now() 

year = now.strftime("%Y")
month = now.strftime("%m")
day = now.strftime("%d")
time = now.strftime("%H:%M:%S")

now_str = year + "-" + month + "-" + day + "_" + time

now_str = now_str.replace(":", "-")

while True:

    term = input('Which term would you like to search for?\n')

    # Search the words of interest
    search_list = models.search_similar_words(term, dictionary, tfidf_model, lsi_model, index, lang)

    files_found = [files[i[0]] for i in search_list[0] if i[1] >= THRESHOLD]
    similarity = [i[1] for i in search_list[0] if i[1] >= THRESHOLD]

    print("The concept appears in the following documents:")
    for n in range(len(files_found)):
        print("{} with a similarity of {}".format(files_found[n], similarity[n]))
            
    csv = input('\nWould you like to store the results in a .csv file? (Y/N)\n')        
    if csv == "Y" or csv == "y":
        file_name = "search_{}_{}.csv".format(term, now_str)
        files_df = pd.DataFrame({"File": files_found, "Similarity": similarity, "Search": term})
        files_df.to_csv("./Searches/" + file_name, index=False)
        print("The results were exported in a file named {}".format(file_name))

    decision = input('\nWould you like to search another term? (Y/N)\n')
    if decision == "N" or decision == "n":
        break



