# sentenceBERT
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime 
import argparse 
import re 
from tqdm import tqdm
import json

pd.set_option('display.max_colwidth', None)

# read stuff
d = pd.read_csv("../data/raw/elaborating_possibilities_study.csv")

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

# add background and endstate
d["background"] = "Robert grew up in a poor neighborhood where criminality and bullying was normal. Robert was bullied and always felt like an outsider, being more interested in literature and theater than in money and girls. Robert currently has a job that he is not passionate about, but he really dreams about making it as an actor."
d["endstate"] = "Robert is now making a living as an actor."
d["ID"] = ["ID0", "ID1", "ID2"]
d = d.drop(columns = {"timestamp", "comments"})

# reorder columns
d = d[[
    "ID",
    "background",
    "action_1",
    "consequence_1",
    "action_2",
    "consequence_2",
    "action_3",
    "consequence_3",
    "endstate"]]

# create dictionary
dct = d.set_index('ID').T.to_dict() # create nested dictionary

# basic cleaning (but now does not apply to the df)
for sub_dct in dct.keys(): # looping over sub_dictionaries
    dct[sub_dct] = {k:cleaning(v) for (k,v) in dct[sub_dct].items()}

# save dct
data = json.dumps(dct) # encoding
with open("../data/processed/pilot.json","w") as f:
  f.write(data)

# save wide
d = pd.DataFrame.from_dict(dct).T
d["ID"] = d.index
d.to_csv("../data/processed/pilot_wide.csv", index=False)

# save long
# reshape
d = pd.melt(
  d, 
  id_vars = "ID",
  value_vars = [
    "background",
    "action_1",
    "consequence_1",
    "action_2",
    "consequence_2",
    "action_3",
    "consequence_3",
    "endstate"
  ])

d.to_csv('../data/processed/pilot_long.csv', index=False)