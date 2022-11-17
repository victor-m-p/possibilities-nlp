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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


####### actual thing now #######
## do 10x10 and then a heatmap ## 
def read_sentences(infile):
    with open(f"{infile}", "r") as f: 
        sentences = f.read().splitlines() 
    return sentences 


## run all pair-wise similarities
### really should only do this for one way (i.e. we do double work here)
def compute_distance(embeddings, distance_fun, n_decimals): 
    vals_lst = [distance_fun(
        [embeddings[x]],
        embeddings[0:])
        for x in range(len(embeddings))]
    vals_concat = np.concatenate(vals_lst)
    vals_rounded = vals_concat.round(n_decimals)
    return vals_rounded 

def plot_hm(sentences, vals, outpath, filename, metric, figsize = (8, 8)):

    ## plot 
    fig, ax = plt.subplots(figsize = figsize)
    im = ax.imshow(vals)

    # show ticks and labels 
    ax.set_xticks(np.arange(len(sentences)), labels = sentences)
    ax.set_yticks(np.arange(len(sentences)), labels = sentences)

    # rotate ticks and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode = "anchor")

    # loop over elements and create annotations
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            text = ax.text(j, i, vals[i, j],
                        ha="center", va="center", color="w", size = 10)

    # title
    plt.suptitle(f"{metric} on sentenceBERT embeddings", size = 15)
    plt.tight_layout()

    # save 
    plt.savefig(f"{outpath}/hm_{filename}.png") # should be pdf

# main
def main(infile, outpath, metric): 
    '''
    metric: 
    * cosine_similarity
    * euclidean_distances
    '''

    ## import sentenceBERT model & instantiate
    print("instantiating model")
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    ## read sentences 
    print("reading sentences")
    sentences = read_sentences(infile)

    ## generate emeddings
    print("sentence embeddings")
    sentence_embeddings = model.encode(sentences)

    ## compute distance
    print("computing distances")
    if metric == "cosine_similarity":
        vals = compute_distance(sentence_embeddings, cosine_similarity, 2)
    else: 
        vals = compute_distance(sentence_embeddings, euclidean_distances, 2)

    ## plot 
    print("plotting heatmap")
    input_name = re.search(r"sentences/(.*?).txt", infile)[1] 
    outname = input_name + "_" + metric
    plot_hm(sentences, vals, outpath, outname, metric) # figsize = (10, 10) baseline

# setup 
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--infile", required=True, type=str, help="file to process (csv)")
    ap.add_argument("-o", "--outpath", required=True, type=str, help='path to folder for saving output files (txt)')
    ap.add_argument('-m', "--metric", required=True, type=str, help="distance metric")
    args = vars(ap.parse_args())
    main(
        infile = args["infile"], 
        outpath = args["outpath"],
        metric = args["metric"])