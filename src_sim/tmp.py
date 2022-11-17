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

pd.set_option('display.max_colwidth', None)

# read stuff
d = pd.read_csv("../data/elaborating_possibilities_study.csv")
background = "Robert grew up in a poor neighborhood where criminality and bullying was normal. Robert was bullied and always felt like an outsider, being more interested in literature and theater than in money and girls. Robert currently has a job that he is not passionate about, but he really dreams about making it as an actor."
endstate = "Robert is now making a living as an actor."

# normalize (see: https://github.com/victor-m-p/reform-psychology/blob/main/twitter/nlp/preprocessing/preprocess_by_tweet.py)
def cleaning(sentence): 
    sentence = re.sub("[^0-9a-zA-Z]+", " ", sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.strip()
    sentence = sentence.lower()
    return sentence
    
# clean stuff 
d.rename(
    columns = {
        "Timestamp": "timestamp",
        "Your suggested action 1": "action_1",
        "What happens after action 1": "consequence_1",
        "What you suggest Robert does next (action 2)": "action_2",
        "What happens after action 2": "consequence_2",
        "What you suggest Robert does next (action 3)": "action_3",
        "What happens after action 3": "consequence_3",
        "Any Comments or Suggestions for Victor and I?": "comments"
    },
    inplace = True
)

# sentence similarity
# outdated model 
model = SentenceTransformer('bert-base-nli-mean-tokens')

# cosine similarity (check native approaches to BERT?)
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

## all action 1 compared to background 
def across_vs_baseline(generation_step, baseline, d): 
    '''
    generation_step: column in d (e.g. action_1)
    baseline: string (e.g. background)
    d: dataframe of responses
    '''
    dct = {}
    baseline = cleaning(baseline)
    for i in range(len(d)):
        test_sentence = cleaning(d[generation_step][i])
        sim = cosine(model.encode(baseline), model.encode(test_sentence))
        dct[(test_sentence, baseline)] = sim
    return dct

## test across different columns 
relevant_columns = [
    "action_1",
    "consequence_1",
    "action_2",
    "consequence_2",
    "action_3",
    "consequence_3"
]

## try across columns
lst_background = []
for column in relevant_columns: 
    dct_tmp = across_vs_baseline(column, endstate, d)
    lst_background.append(dct_tmp)

for dictionary in lst_background: 
    df = pd.Series(dictionary).reset_index()   
    df.columns = ['w1', 'w2', 'sim']
    mean_val = df["sim"].mean()
    print(mean_val)
    
## sanity check (first and third more similar than second guy)
### we do in fact observe this ### 
lst_main = []
for col in relevant_columns:
    lst = list(d[col])
    dct = {}
    for sentence in lst: 
        #print(f"{sentence}")
        for query in lst: 
            #print(f"{query}")
            sim = cosine(model.encode(sentence), model.encode(query))
            dct[(sentence, query)] = sim
    lst_main.append(dct)

df = pd.Series(lst_main[5]).reset_index()   
df.columns = ['w1', 'w2', 'sim']
df.head(10)

## how dissimilar is each step to the previous step ## 
### not quite as good as the "directional" idea, but getting somewhere ###

## "probability" of sentence given previous. 
## "probability" of sentence (broadly?, normalized?) 
## "call-backs"?

# observations: 
## Does not look like people really split it into (action, consequence) pairs?
## I think we SHOULD go for something like (event 1, event 2, ..., event x).

# useful comments: 
## back-and-forth (which we also talked about) rather than linear
## can we capture this somehow (is a matter of design I guess). 

#### can we see that (1) and (3) are more similar to each other than (2) to any of them?
