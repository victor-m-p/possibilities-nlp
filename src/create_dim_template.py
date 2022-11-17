from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import itertools 
from tqdm import tqdm

# morality (see J. Evans)
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

# no backing for this unfortunately
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

# read stuff
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
d = pd.read_csv("../data/chris_clean/CN_Study2_actions.csv")

# Mean "morality" vector
## subtract (probably not best)
def subtract(word_tuple): 
    x, y = word_tuple
    return model.encode(x) - model.encode(y)

## mean vector (probably not best)
lst = []
for i in morality_pairs: 
    difference = subtract(i)
    lst.append(difference)
mean_morality = np.mean(lst, axis=0)

mean_morality

# test sentences 
moral_sentences = [
    "i sacrificed myself to save a baby", 
    "help the old lady cross the street",
    "i bought coke and milk",
    "cheat them to think I did not do it",
    "steal from the counter",
    "then i killed my best friend"
]

## does capture something, but not enough
moral_embeddings = model.encode(moral_sentences, convert_to_tensor=True)
dist = util.cos_sim(mean_morality, moral_embeddings)[0] # dot score just scaled

#### test it quickly #### 
d = pd.read_csv("../data/chris_clean/CN_Study2_outcomes.csv")
d.head(5)
list(d["n1"])

d_overall = []
for col in ["p1", "p2", "n1", "n2"]: 
    
    # corpus embedding
    corpus = list(d[col])
    corpus_embedding = model.encode(corpus, convert_to_tensor=True)
    # see documentation: https://www.sbert.net/examples/applications/semantic-search/README.html
    hits = util.semantic_search(mean_morality, corpus_embedding, top_k = len(corpus))
    
    # this we should do smarter
    hits_unlst = [item for sublist in hits for item in sublist]
    d_emb = pd.DataFrame(hits_unlst)
    d_emb = d_emb.sort_values("corpus_id")
    d_emb["text"] = corpus
    d_emb["type"] = col
    d_overall.append(d_emb)

d_test = pd.concat(d_overall)
d_test.groupby("type")["score"].mean() # positive negative morality

d_test.sort_values("score").head(5) # negative morality
d_test.sort_values("score", ascending=False).head(5) # positive morality

# no difference between first, second. 
sns.displot(
    d_test, 
    x="score", 
    hue="type", 
    kind="kde", 
    fill=True)
plt.suptitle("morality classification")
plt.savefig('../fig/morality.png')