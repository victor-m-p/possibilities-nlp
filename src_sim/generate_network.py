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

# load preprocessed data
with open("../data/processed/pilot.json","r") as f:
  data = f.read()
dct = json.loads(data) # decoding
d = pd.read_csv("../data/processed/pilot_long.csv")

## import sentenceBERT model & instantiate
model = SentenceTransformer('bert-base-nli-mean-tokens')

## itertools. 
def rSubset(arr, r):
    return set(list(combinations(arr, r))) 

# getting all combinations of sentences. 
sentences = list(d["value"].unique()) # unique avoids (endstate, endstate)
sentence_pairs = rSubset(sentences, 2)

# all of these combinations we are going to be embedding. 
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# 1 minute (could explode, perhaps we can do something smarter). 
def compute_cosine_sim(sentence_pairs, cosine, model): 
    dct = {}
    for i, j in tqdm(sentence_pairs): 
        sim = cosine(model.encode(i), model.encode(j))
        dct[(i, j)] = sim
    return dct

cosine_dct = compute_cosine_sim(sentence_pairs, cosine, model)
df = pd.Series(cosine_dct).reset_index()   
df.columns = ['w1', 'w2', 'sim']

# save edgelist 
df.to_csv('../data/final/edgelist.csv', index = False)