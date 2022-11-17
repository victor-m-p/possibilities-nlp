from turtle import color
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import re 

# read 
d_within = pd.read_csv("../data/final/pca_indiv.csv")
d_across = pd.read_csv("../data/final/pca_across.csv")

# plot single cases (within)
def plot_single_pca(d, ID, condition):
    '''
    d: pd.Dataframe
    ID: string
    condition: string (for filename only)
    '''
    fig, ax = plt.subplots(1, figsize = (10, 5), dpi = 300)
    d = d[d["ID"] == ID]
    sentences = list(d["value"])
    x = list(d["dim1"])
    y = list(d["dim2"])
    x_max = max(x)
    y_grid = np.linspace(-0.1, -0.8, len(y))
    plt.scatter(x, y)
    plt.margins(x=0.2, y=0.2)
    sentences = [re.sub("(.{100})", "\\1\n", x, 0, re.DOTALL) for x in sentences]
    for i, label in enumerate(sentences):
        plt.annotate(f"{i}", (x[i], y[i]+0.2), ha='center')
        plt.text(0.15, y_grid[i], f"{i}: {sentences[i]}", va = 'top', fontsize=10, transform=plt.gcf().transFigure)
    plt.axis('off')
    plt.savefig(f"../fig/pca_single_{condition}_{ID}.jpg", bbox_inches='tight') 

# generate all the plots 
for ID in d_within["ID"].unique():
    plot_single_pca(d_within, ID, "within")
    plot_single_pca(d_across, ID, "across")
    
# plot together (without text I guess). 
# make the plot nice ... 
def plot_all(d):
    '''
    d: pd.Dataframe
    '''
    fig, ax = plt.subplots(1, figsize = (10, 5), dpi = 300)
    clrs = ["tab:blue", "tab:orange", "tab:green"]
    xlab = np.linspace(0.3, 0.6, 3)
    for i, ID in enumerate(d["ID"].unique()): 
        dx = d[d["ID"] == ID]
        sentences = list(dx["value"])
        x = list(dx["dim1"])
        y = list(dx["dim2"])
        x_max = max(x)
        plt.scatter(x, y, color = clrs[i])
        plt.plot(x, y, color = clrs[i])
        plt.margins(x=0.2, y=0.2)
        sentences = [re.sub("(.{100})", "\\1\n", x, 0, re.DOTALL) for x in sentences]
        for j, label in enumerate(sentences):
            plt.annotate(f"{j}", (x[j], y[j]+0.3), ha='center', color=clrs[i])
        plt.text(xlab[i], 0, f"ID: {i}", fontsize=10, color = clrs[i], transform=plt.gcf().transFigure)
    plt.axis('off')
    plt.savefig(f"../fig/pca_across.jpg", bbox_inches='tight') 

plot_all(d_across)

# plot with crazy text
def plot_all_text(d):
    '''
    d: pd.Dataframe
    '''
    fig, ax = plt.subplots(1, figsize = (2, 1), dpi = 300)
    clrs = ["tab:blue", "tab:orange", "tab:green"]
    x_grid = np.linspace(0.15, 0.6, 3)
    n_unique = d["variable"].unique()
    y_grid = np.linspace(-0.4, -2, len(n_unique)-1)
    y_grid = np.append(0.1, y_grid)
    for i, ID in enumerate(d["ID"].unique()): 
        dx = d[d["ID"] == ID]
        sentences = list(dx["value"])
        x = list(dx["dim1"])
        y = list(dx["dim2"])
        x_max = max(x)
        plt.scatter(x, y, color = clrs[i], s=0.5)
        plt.plot(x, y, color = clrs[i], linewidth=0.3)
        plt.margins(x=0.2, y=0.2)
        sentences = [re.sub("(.{27})", "\\1\n", x, 0, re.DOTALL) for x in sentences]
        for j, label in enumerate(sentences):
            plt.annotate(f"{j}", (x[j], y[j]+0.3), ha='center', fontsize = 4,color=clrs[i])
            plt.text(x_grid[i], y_grid[j], f"{j}: \n{sentences[j]}", va = 'top', fontsize=2, color = clrs[i], transform=plt.gcf().transFigure)
    plt.axis('off')
    plt.savefig(f"../fig/pca_across_text.jpg", bbox_inches='tight') 

plot_all_text(d_across)