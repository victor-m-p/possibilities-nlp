# sentenceBERT
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime 
import argparse 
import re 
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA 
import json
from itertools import combinations

# dimensions
## morality (Evans, The Geometry of Culture)
morality_pairs = [
    ("good", "evil"),
    ("moral", "immoral"),
    ("good", "bad"),
    ("honest", "dishonest"),
    ("virtuous", "sinful"),
    ("virtue", "vice"),
    ("righteous", "wicked"),
    ("chaste", "transgressive"),
    ("principled", "unprincipled"),
    ("unquestionable", "questionable"),
    ("noble", "nefarious"),
    ("uncorrupt", "corrupt"),
    ("scrupulous", "unscrupulous"),
    ("altruistic", "selfish"),
    ("chivalrous", "knavish"),
    ("honest", "crooked"),
    ("commendable", "reprehensible"),
    ("pure", "impure"),
    ("dignified", "undignified"),
    ("holy", "unholy"),
    ("valiant", "fiendish"),
    ("upstanding", "villanous"),
    ("guiltless", "guilty"),
    ("decent", "indecent"),
    ("chaste", "unsavory"),
    ("righteous", "odious"),
    ("ethical", "unethical")
]

# no list from them unfortunately
probability_pairs = [
    ("easy", "difficult"),
    ("easy", "hard"),
    ("probable", "improbable"),
    ("possible", "impossible"),
    ("expected", "unexpected"),
    ("normal", "unusual"),
    ("normal", "rare"),
    ("expected", "lucky")
]

## import sentenceBERT model & instantiate
model = SentenceTransformer('bert-base-nli-mean-tokens')

# helper functions
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def subtract(word_tuple): 
    x, y = word_tuple
    return model.encode(x) - model.encode(y)

### test morality ###
# generate morality vector
lst = []
for i in morality_pairs: 
    difference = subtract(i)
    lst.append(difference)
mean_morality = np.mean(lst, axis=0)

# test sentences 
moral_sentences = [
    "i sacrificed myself to save a baby", 
    "help the old lady cross the street",
    "i bought sugar and milk",
    "cheat them to think I did not do it",
    "steal from the counter",
    "then i killed my best friend"
]

dct = {}
for i in moral_sentences: 
    dist = cosine(mean_morality, model.encode(i))
    dct[i] = dist

dct

# notes
## thinks that steal is more immoral than kill
## having "hot sex" and "buying milk" is the same (which is fine, but ...)

### test probability ###
# generate probability vector
lst = []
for i in probability_pairs: 
    difference = subtract(i)
    lst.append(difference)
mean_probability = np.mean(lst, axis=0)

# test sentences 
probability_sentences = [
    "win in lotto", 
    "take a loan",
    "",
    ""
]

dct = {}
for i in probability_sentences: 
    dist = cosine(mean_probability, model.encode(i))
    dct[i] = dist

dct