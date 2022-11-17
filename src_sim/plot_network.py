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
import networkx as nx 
from collections import defaultdict
pd.set_option('display.max_colwidth', None)

# read
d = pd.read_csv("../data/processed/pilot_long.csv")
edgelist = pd.read_csv("../data/final/edgelist.csv")

# recoding of dataframe
## better to display for now 
d = d.replace({
  'variable': {
    "background": 0,
    "action_1": 1,
    "consequence_1": 2,
    "action_2": 3,
    "consequence_2": 4,
    "action_3": 5,
    "consequence_3": 6,
    "endstate": 7
  }})

## recode start and end value 
d.loc[d.variable == 0, "ID"] = "start"
d.loc[d.variable == 7, "ID"] = "end"
d = d.drop_duplicates()

## prepare stuff
## color 
conditions = [
    (d['ID'] == "ID0"),
    (d['ID'] == "ID1"),
    (d['ID'] == "ID2"),
    (d['ID'] == "start") | (d['ID'] == "end")
    ]

values = ["tab:blue", "tab:orange", "tab:green", "tab:grey"]

d["col"] = np.select(conditions, values)

attrs = d.set_index('value').T.to_dict()

## create network 
edgelist = edgelist.rename(columns = {'sim': 'weight'})

G = nx.from_pandas_edgelist(
    edgelist,
    source = "w1",
    target = "w2",
    edge_attr = ["weight"]
)

## add data & extract
nx.set_node_attributes(G, attrs)
nodelst = list(G.nodes())
nodeattr_color = list(nx.get_node_attributes(G, "col").values())

label_dct = defaultdict()
for node, attr in G.nodes(data=True): 
  label_dct[node] = attr.get('variable')
label_dct = dict(label_dct)

fig, ax = plt.subplots(figsize=(8, 8), dpi=300, facecolor='w', edgecolor='k')
plt.axis("off")

seed = 13 
pos = nx.spring_layout(
    G = G,
    seed = seed)

# nodes
nx.draw_networkx_nodes(
        G, 
        pos, 
        nodelist = nodelst, 
        edgecolors = "black",
        node_color = nodeattr_color,
        linewidths = 0.5)

# labels
nx.draw_networkx_labels(
    G, 
    pos, 
    labels = label_dct,
    font_size = 10) 

