# sentenceBERT
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

## import sentenceBERT model
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

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
sentence_embeddings = model.encode(sentences) # not used at the moment.

# cosine similarity (check native approaches to BERT?)
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

### check something 
sentences = [
    "I love delicious food",
    "Alien Dominguez kill kitten"
]

## run all pair-wise similarities
### really should only do this for one way (i.e. we do double work here)

dct = {}
for sentence in sentences: 
    print(f"sentence: {sentence}")
    for query in sentences: 
        print(f"query: {query}")
        sim = cosine(model.encode(sentence), model.encode(query))
        dct[(sentence, query)] = sim

## dictionary to pandas
df = pd.Series(dct).reset_index()   
df.columns = ['w1', 'w2', 'sim']
df.head(5)

## reshape pandas and prepare heatmap
df_unstacked = df.set_index(['w1', 'w2']).unstack()
df_unstacked.columns = df_unstacked.columns.droplevel()
x = list(df_unstacked.columns)
y = list(df_unstacked.index)
vals = df_unstacked.to_numpy()
vals = vals.round(decimals = 2) # round decimals

## plot 
fig, ax = plt.subplots(figsize = (10, 10))
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
                       ha="center", va="center", color="w")

tst = [x for x in range(len(sentence_embeddings))]
tst

### NB: better set-up ###
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

def compute_distance(embeddings, distance_fun, n_decimals): 
    vals_lst = [distance_fun(
        [embeddings[x]],
        embeddings[0:])
        for x in range(len(embeddings))]
    vals_concat = np.concatenate(vals_lst)
    vals_rounded = vals_concat.round(n_decimals)
    return vals_rounded 

tst = compute_distance(sentence_embeddings, euclidean_distances, 3)


pd.DataFrame(sentences)


vals_lst = [cosine_similarity(
    [sentence_embeddings[x]],
    sentence_embeddings[0:]) 
    for x in range(len(sentence_embeddings))]

vals_concat = np.concatenate(vals_lst, axis=0 )
vals_rounded = vals_concat.round(2)


for x in range(len(sentence_embeddings)): 
    print(x) 
    vals = cosine_similarity(
        [sentence_embeddings[x]],
        sentence_embeddings[0:]
    )
    print(vals)


sentence_embeddings = model.encode(sentences)
cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[0:]
)

