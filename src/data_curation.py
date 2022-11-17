import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import os
os.getcwd()

# read data
d = pd.read_csv("../data/chris_raw/CN_Study_2_for_Victor.csv", encoding_errors='ignore')

# sort data
## global variables
context = ["Alc", "Parent", "Gym", "CEO", "Supply"]
context_abb = ["A", "P", "G", "C", "S"]

## helper function for renaming
def rename_cols(d, list_old, list_new):
    dct = {k:v for (k,v) in zip(list_old, list_new)}
    d = d.rename(columns=dct)
    return d

def r_text(d, pattern1, pattern_new, context): 
    c_old = [f"{pattern1}{x}" for x in context]
    c_new = [f"{pattern_new}{x}" for x in context]
    d_new = rename_cols(d, c_old, c_new)
    return d_new, c_new
    
def r_rt(d, pattern1, pattern2, pattern_new, context, context_abb): 
    c_old = [f"{pattern1}{x}{pattern2}" for x in context_abb]
    c_new = [f"{pattern_new}{x}" for x in context]
    d_new = rename_cols(d, c_old, c_new)
    return d_new, c_new

def r_double(d, pattern1, pattern2, pattern_new, context): 
    c_old = [f"{pattern1}{x}{pattern2}" for x in context]
    c_new = [f"{pattern_new}{x}" for x in context]
    d_new = rename_cols(d, c_old, c_new)
    return d_new, c_new

def r_out_2(d, pattern1, pattern2, pattern_fix, pattern_new, context): 
    c_old = [f"{x}{pattern1}" if "Alc" in x or "Parent" in x else f"{x}{pattern2}" for x in context] # get colnames
    c_old = [f"{pattern_fix}" if "CEO" in x else x for x in c_old] # fix inconsistency
    c_new = [f"{pattern_new}{x}" for x in context] # new colnames
    d_new = rename_cols(d, c_old, c_new) # rename
    return d_new, c_new

# renaming 
## rt
d, c_rt = r_rt(d, "G", "T_Page Submit", "rt_", context, context_abb)
## text (solutions)
d, c_txt = r_text(d, "Gen", "text_", context)
## fix ratings 
d, c_r1 = r_double(d, "Gen", "1", "r1_", context)
d, c_r2 = r_double(d, "Gen", "2", "r2_", context)

### using the index ###
d1 = d[d["Indicator"] == 1]
d0 = d[d["Indicator"] == 0]

## fix outcomes (v1)
d1, c_p1_1 = r_double(d1, "Gen", "Pos1", "p1_", context)
d1, c_p2_1 = r_double(d1, "Gen", "Pos2", "p2_", context)
d1, c_n1_1 = r_double(d1, "Gen", "Neg1", "n1_", context)
d1, c_n2_1 = r_double(d1, "Gen", "Neg2", "n2_", context)

## fix outcome (v2)
d0, c_p1_0 = r_out_2(d0, "NegCons1", "Neg1", "CEOneg1", "n1_", context)
d0, c_p2_0 = r_out_2(d0, "NegCons2", "Neg2", "CEOneg2", "n2_", context)
d0, c_n1_0 = r_out_2(d0, "PosCons1", "Pos1", "CEOpos1", "p1_", context)
d0, c_n2_0 = r_out_2(d0, "PosCons2", "Pos2", "CEOpos2", "p2_", context)

# wide to long format 
## helper function
def wide_long(d, cols, id, stub):
    d = d[cols]
    d_long = pd.wide_to_long(
        d, 
        stubnames=stub,
        i=id,
        j="condition",
        sep="_",
        suffix=".*").reset_index()
    return d_long 

d0_long = wide_long(
    d = d0,
    cols = ["RandomID", "Indicator"] + c_p1_0 + c_p2_0 + c_n1_0 + c_n2_0,
    id = "RandomID",
    stub = ["p1", "p2", "n1", "n2"]
)

d1_long = wide_long(
    d = d1,
    cols = ["RandomID", "Indicator"] + c_p1_1 + c_p2_1 + c_n1_1 + c_n2_1,
    id = "RandomID",
    stub = ["p1", "p2", "n1", "n2"]
)

d_rt_long = wide_long(d, ["RandomID"] + c_rt, "RandomID", "rt")
d_text_long = wide_long(d, ["RandomID"] + c_txt, "RandomID", "text")
d_rating_1_long = wide_long(d, ["RandomID"] + c_r1, "RandomID", "r1")
d_rating_2_long = wide_long(d, ["RandomID"] + c_r2, "RandomID", "r2")

# combine data
d_out_long = pd.concat([d0_long, d1_long])

# bind together
dfs = [
    d_out_long,
    d_rt_long, 
    d_text_long, 
    d_rating_1_long, 
    d_rating_2_long]

d_main = reduce(lambda left, right: pd.merge(left, right, on=['RandomID', 'condition']), dfs)

# save for now
d_main.to_csv("../data/chris_clean/CN_Study2_clean.csv", index=False)

## apply sentiment analysis
## check whether we can predict shift in rating based on features of outcomes 
## ask where the "free" response is (i.e. as opposed to n1, n2, p1, p2...)
## put this stuff on github (without the data and secret repo) -- run test first to check that it works without putting up the data...

