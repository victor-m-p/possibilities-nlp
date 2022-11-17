'''
VMP 2022-06-29: 
usage: 
python hm.py -i sentences/basics.txt -o fig
'''

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

####### actual thing now #######
## do 10x10 and then a heatmap ## 
def read_sentences(infile):
    with open(f"{infile}", "r") as f: 
        sentences = f.read().splitlines() 
    return sentences 

# cosine similarity (check native approaches to BERT?)
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

## run all pair-wise similarities
### really should only do this for one way (i.e. we do double work here)
def compute_cosine_sim(sentences, cosine, model): 
    dct = {}
    for sentence in tqdm(sentences): 
        for query in sentences: 
            sim = cosine(model.encode(sentence), model.encode(query))
            dct[(sentence, query)] = sim
    return dct

## dictionary to pandas
def reshape_dct(dct, decimals = 2):

    # to pandas 
    df = pd.Series(dct).reset_index()   
    df.columns = ['w1', 'w2', 'sim']

    ## reshape pandas and prepare heatmap
    df_unstacked = df.set_index(['w1', 'w2']).unstack()
    df_unstacked.columns = df_unstacked.columns.droplevel()
    x = list(df_unstacked.columns)
    y = list(df_unstacked.index)
    vals = df_unstacked.to_numpy()
    vals = vals.round(decimals = decimals) # round decimals

    return df

def plot_hm(x, y, vals, outpath, filename, figsize = (8, 8)):

    ## plot 
    fig, ax = plt.subplots(figsize = figsize)
    im = ax.imshow(vals)

    # show ticks and labels 
    ax.set_xticks(np.arange(len(x)), labels = x)
    ax.set_yticks(np.arange(len(y)), labels = y)

    # rotate ticks and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode = "anchor")

    # loop over elements and create annotations
    for i in range(len(y)):
        for j in range(len(x)):
            text = ax.text(j, i, vals[i, j],
                        ha="center", va="center", color="w", size = 10)

    # title
    plt.suptitle(f"Cosine Distance on sentenceBERT embeddings", size = 15)
    plt.tight_layout()

    # save 
    plt.savefig(f"{outpath}/hm_{filename}.png") # should be pdf


infile = "/home/victor/sentence-sim/sentences/basics.txt"
## import sentenceBERT model & instantiate
print("instantiating model")
model = SentenceTransformer('bert-base-nli-mean-tokens')

## read sentences 
print("reading sentences")
sentences = read_sentences(infile)

## compute distance 
print("computing cosine distance with sentenceBERT")
dct = compute_cosine_sim(sentences, cosine, model)

## reshape
print(f"reshape results")
df = reshape_dct(dct) # decimals = 2 baseline
df = df.rename(columns = {'sim': 'weight'})

##### network #####
## remove duplicates
import networkx as nx 
from collections import defaultdict

G = nx.from_pandas_edgelist(
    df,
    source = "w1",
    target = "w2",
    edge_attr = ["weight"]
)

node_list = list(G.nodes())
df = df[df["w1"] < df["w2"]]
n_labels = len(sentences) # set to 10

# pretty stupid right now # 
dct_labels = defaultdict()
for num, ele in enumerate(node_list): 
    dct_labels[ele] = ele
dct_labels = dict(dct_labels)

#### all values accepted 
def plot_net(G, cutoff): 

    edgeattr_weight = nx.get_edge_attributes(G, "weight") ## need to sort here as well
    edge_width_list = list(edgeattr_weight.values())
    edge_width_list = [x if x > cutoff else 0 for x in edge_width_list]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, facecolor='w', edgecolor='k')
    plt.axis("off")

    seed = 13 # good: 13, 15 
    pos = nx.spring_layout(
        G = G,
        k = None,
        iterations = 500,
        seed = seed)

    nx.draw_networkx_nodes(
            G, 
            pos, 
            nodelist = node_list, 
            #node_size = node_size, 
            #node_color = node_color_list,
            edgecolors = "black",
            linewidths = 0.5)

    nx.draw_networkx_edges(
        G, 
        pos, 
        #edgelist = edgelst, 
        width = edge_width_list, 
        alpha = 1,
        #edge_color = edge_color_list
        ) 

    nx.draw_networkx_labels(
        G, 
        pos, 
        labels = dct_labels,
        font_size = 10) 

    plt.savefig(f"/home/victor/sentence-sim/fig/cutoff_{cutoff}.png")

plot_net(G, 0)
df["weight"].mean()