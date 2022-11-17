# sentenceBERT
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import itertools 
from tqdm import tqdm

# https://www.sbert.net/docs/pretrained_models.html
## best performance: "all-mpnet-base-v2"
## good performance and fast: "all-MiniLM-L6-v2"
## Srinivasan: "multilingual-MiniLM-L12-v2" (not best)

# util
## util.dot_score() -- dot product
## util.cos_sim() -- cosine similarity

# semantic search
## https://www.sbert.net/examples/applications/semantic-search/README.html
## I am not entirely sure that this is what we want. I think we just want distance..?

""" example:
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query_embedding = model.encode('How big is London')
passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                  'London is known for its finacial district'])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))
"""

# preparation
## import model 
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1") # the good one 
d = pd.read_csv("../data/chris_clean/CN_Study2_actions.csv")

## query embedding # https://www.sbert.net/examples/applications/semantic-search/README.html
queries = {
    "Alc": "Nearly 2,000 students die due to alcohol-related injuries in the US each year. Imagine that you are the dean of a college where heavy drinking is becoming a problem on campus. What policy would you propose to best promote a safer alcohol climate at your school?",
    "Parent": "Imagine that you are a parent of a sophomore in high school who has suddenly decided to stop doing his homework. As his parent, you are concerned and want to get him back on track. What would you try to do to get him to do his homework?",
    "Gym": "Imagine that you are the owner of a local gym that charges $40 per month to members. While members can quit in any given month, you are excited to announce you've recently gained your 1000th member. In celebration, you plan to upgrade the gym's equipment next year. How would you get the money to fund the budget for this new equipment?",
    "CEO": "Imagine that you are the CEO of a Fortune 500 corporation which has been accused of racial discrimination. What would you do to improve your public image?",
    "Supply": "Imagine that someone in your office has been stealing supplies (i.e. pens and paper). Your boss has put you in charge of finding a solution to prevent this theft from occurring. Your boss likes to provide these supplies because she thinks it boosts productivity, so ceasing to offer supplies is not an option. What solution would you propose to your boss to solve this problem?"
}

# combinations 
t = ["Alc", "Parent", "Gym", "CEO", "Supply"]
c1 = list(set(itertools.combinations(t, 2)))
c2 = [(x, x) for x in t]
c3 = c1 + c2

### 1. check that we can tell which vignette answers are to ### 
d_overall = []
for key_pair in tqdm(c3): 
    # unpack
    key_query, key_corpus = key_pair 
    # query embedding 
    query_embedding = model.encode(queries.get(key_query), convert_to_tensor=True)
    # corpus embedding
    d_tmp = d[d["condition"] == key_corpus][["text"]]
    corpus = list(d_tmp["text"])
    corpus_embedding = model.encode(corpus, convert_to_tensor=True)
    # see documentation: https://www.sbert.net/examples/applications/semantic-search/README.html
    hits = util.semantic_search(query_embedding, corpus_embedding, top_k = len(corpus))
    
    # this we should do smarter
    hits_unlst = [item for sublist in hits for item in sublist]
    d_emb = pd.DataFrame(hits_unlst)
    d_emb = d_emb.sort_values("corpus_id")
    d_emb["text"] = corpus
    d_emb["query"] = key_query
    d_emb["corpus"] = key_corpus
    d_overall.append(d_emb)

# this works for sure
d_emb = pd.concat(d_overall)
d_emb.groupby(['query', 'corpus'])['score'].mean() # much higher for congruent

# check whether "relevance" predicts "goodness"
d_emb_cong = d_emb[d_emb["query"] == d_emb["corpus"]] # 901 vs. 905...
d_emb_cong = d_emb_cong[["score", "text", "query"]]
d_emb_cong = d_emb_cong.rename(columns = {"query": "condition"})
d_w_cong = d_emb_cong.merge(d, on = ["text", "condition"])

# correlation (not a strong predictor across)
## okay, need to make the nice plot 
d_w_cong = d_w_cong.assign(delta = lambda x: x["r2"] - x["r1"])

sns.lmplot(
    data = d_w_cong,
    x = "score",
    y = "r1",
    hue = "condition"
)
plt.suptitle("relevance (r1)")
plt.savefig('../fig/relevance_r1.png')

sns.lmplot(
    data = d_w_cong,
    x = "score",
    y = "r2",
    hue = "condition"
)
plt.suptitle('relevance (r2)')
plt.savefig('../fig/relevance_r2.png')

sns.lmplot(
    data = d_w_cong,
    x = "score",
    y = "delta",
    hue = "condition"
)
plt.suptitle('relevance (delta: r2-r1)')
plt.savefig('../fig/relevance_delta.png')

# plot distribution (i.e. distance). 
