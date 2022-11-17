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

# load preprocessed data
with open("../data/processed/pilot.json","r") as f:
  data = f.read()
dct = json.loads(data) # decoding
d = pd.read_csv("../data/processed/pilot_long.csv")

## import sentenceBERT model & instantiate
model = SentenceTransformer('bert-base-nli-mean-tokens')

## PCA (NB: test whether 3 components is significantly better, we can do 3D visualization)
### individually
def get_single_embeddings(dct, ID):
  k = list(dct.get(ID).keys())
  v = list(dct.get(ID).values())
  sentence_embeddings = model.encode(v)
  pca = PCA(n_components=2) # is this actually just the same as creting network?
  pca.fit(sentence_embeddings)
  print(pca.explained_variance_ratio_) # 50%
  X = pca.transform(sentence_embeddings)
  d = pd.DataFrame(X, columns = ["dim1", "dim2"])
  d["ID"] = ID
  d["variable"] = k
  return d

pca_indiv = [get_single_embeddings(dct, i) for i in list(dct.keys())]
pca_indiv = pd.concat(pca_indiv)
pca_indiv = pd.merge(pca_indiv, d, on = ["ID", "variable"])
pca_indiv # this has exactly the problem I thought. 

### collectively
def get_collective_embeddings(dct): 
  id_lst = []
  k_lst = []
  v_lst = []
  for sub_dct in dct.keys():
    for k, v in dct.get(sub_dct).items(): 
      k_lst.append(k)
      v_lst.append(v)
      id_lst.append(sub_dct)
  sentence_embeddings = model.encode(v_lst)
  pca = PCA(n_components=2) # 35%
  pca.fit(sentence_embeddings)
  print(pca.explained_variance_ratio_)
  X = pca.transform(sentence_embeddings)
  d = pd.DataFrame(X, columns = ["dim1", "dim2"])
  d["ID"] = id_lst
  d["variable"] = k_lst
  d["value"] = v_lst
  return d 

pca_across = get_collective_embeddings(dct)
pca_across.head(10)

### save both 
pca_indiv.to_csv("../data/final/pca_indiv.csv", index = False)
pca_across.to_csv("../data/final/pca_across.csv", index = False)
