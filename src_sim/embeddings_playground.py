'''
VMP 2022-06-29: 
playing with sentences for CMU project. 
'''

# following: https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/?fbclid=IwAR3KK3koLd7gse1hUqtp4oSJoqnPIUzI_K3aErGPge8g84fss9n6Tr4-Mbs

## imports 
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np

## test sentences 
sentences = [
    "take an uber", 
    "call a cab", 
    "ring a taxi",
    "take the bus",
    "call a friend",
    "I can fly there"]

## tokenize 
from nltk.tokenize import word_tokenize
tokenized_sent = [word_tokenize(sentence) for sentence in sentences]

tokenized_sent

## cosine
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

## Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
tagged_data

## train doc2vec model
model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)

'''
vector_size = Dimensionality of the feature vectors.
window = The maximum distance between the current and predicted word within a sentence.
min_count = Ignores all words with total frequency lower than this.
alpha = The initial learning rate.
'''

## Print model vocabulary
### --- unclear 

## test against other sentence
test_doc = word_tokenize("call a taxi driver".lower())
test_doc_vector = model.infer_vector(test_doc)
model.docvecs.most_similar(positive = [test_doc_vector]) # seems to not get it. 


# sentenceBERT
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embeddings = sbert_model.encode(sentences) # error in their code
query = "call a taxi driver"
query_vec = sbert_model.encode([query])[0]

## better, but we should test it more... ## 
for sent in sentences:
  sim = cosine(query_vec, sbert_model.encode([sent])[0])
  print("Sentence = ", sent, "; similarity = ", sim)

####### actual thing now #######
## do 10x10 and then a heatmap ## 
sentences = [
    "take an uber", 
    "call a cab", 
    "ring a taxi",
    "take the bus",
    "call a friend",
    "I can fly there",
    "steal a car", 
    "steal a taxi", 
    "call the airport",
    "cancel the talk"
]

## encode these sentences 
model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(sentences)

query = "call a taxi driver"
query_vec = sbert_model.encode([query])[0]

## 


## heatmap

# import modules
import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sns
import numpy as np


## specific tests 
### negation
### long sentences vs. short sentences 
### usage of same (non-informative) words 
