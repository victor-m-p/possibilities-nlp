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

# read sentences function
def read_sentences(infile):
    with open(f"{infile}", "r") as f: 
        sentences = f.read().splitlines() 
    return sentences 

## import sentenceBERT model & instantiate
model = SentenceTransformer('bert-base-nli-mean-tokens')

## read sentences 
infile = "sentences/basics.txt" # assuming that we are in /sentence-sim
sentences = read_sentences(infile)

## generate embeddings
sentence_embeddings = model.encode(sentences)

#### PCA 
pca = PCA(n_components=2)
pca.fit(sentence_embeddings)
print(pca.explained_variance_ratio_) # not a lot of explained variance
X = pca.transform(sentence_embeddings)

#### distance
d = pd.DataFrame(X, columns = ["x", "y"])
x = np.array(d["x"])
y = np.array(d["y"])
plt.scatter(x, y)
for i, label in enumerate(sentences):
    plt.annotate(label, (x[i], y[i]))
plt.savefig("/home/victor/sentence-sim/fig/test_pca.png")

for i, label in enumerate(sentences): 
    print(i)
    print(label)
    x[i]
    y[i]