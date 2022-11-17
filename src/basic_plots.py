import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import ptitprince as pt 

d = pd.read_csv("../data/chris_clean/CN_Study2_clean.csv")
d.head(5)

# distribution of ratings 
## melt 
d_r_wide = d[["RandomID", "r1", "r2", "condition", "rt"]]
d_r_long = pd.wide_to_long(
    d_r_wide,
    stubnames=["r"],
    i=["RandomID", "condition",],
    j="rating_n",
    sep=""
).reset_index()
d_r_long['rating_n'] = d_r_long['rating_n'].astype('category')

## overall plot (ignoring condition)
## actually this is pretty fucked for discrete outcome
f, ax = plt.subplots(figsize=(7, 5))
dy="rating_n"; dx="r"; ort="h"; pal = "Set2"

ax=pt.half_violinplot(
    x = dx, 
    y = dy, 
    data = d_r_long, 
    palette = pal, 
    bw = .2, 
    cut = 0.,
    scale = "area", 
    width = .6, 
    inner = None, 
    orient = ort)

ax=sns.stripplot(
    x = dx,
    y = dy,
    data = d_r_long,
    palette = pal,
    edgecolor = "white",
    size = 3,
    jitter = 1,
    zorder = 0,
    orient = ort
)

ax=sns.boxplot(
    x = dx, 
    y = dy, 
    data = d_r_long, 
    color = "black", 
    width = .15, 
    zorder = 10,
    showcaps = True, 
    boxprops = {'facecolor': 'none', "zorder": 10},
    showfliers=True, whiskerprops = {'linewidth': 2, "zorder": 10},
    saturation = 1, 
    orient = ort)

## more concise 
dx = "rating_n"; dy = "r"; ort = "h"; pal = "Set2"; sigma = .2
f, ax = plt.subplots(figsize=(7, 5))

pt.RainCloud(x = dx, y = dy, data = d_r_long, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort)

## rain below
dx = "rating_n"; dy = "r"; ort = "h"; pal = "Set2"; sigma = .2
f, ax = plt.subplots(figsize=(7, 5))

ax=pt.RainCloud(x = dx, y = dy, data = d_r_long, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort, move = .2)

## change orient
# Changing orientation
dx="rating_n"; dy="r"; ort="v"; pal = "Set2"; sigma = .2
f, ax = plt.subplots(figsize=(7, 5))

ax=pt.RainCloud(x = dx, y = dy, data = d_r_long, palette = pal, bw = sigma,
                 width_viol = .5, ax = ax, orient = ort, pointplot=True)


# plot change in ratings
d_r = d[["RandomID", "r1", "r2", "condition", "rt"]]
d_r_long = pd.wide_to_long(
    d_r,
    stubnames=["r"],
    i=["RandomID", "condition",],
    j="rating_n",
    sep=""
).reset_index()
d_r_long['rating_n'] = d_r_long['rating_n'].astype('category')

# I do not observe the change in ratings here. 
# Of course this is extremely crude...
sns.boxplot(
    data=d_r_long,
    x="condition",
    y="r",
    hue="rating_n",
    showmeans=True)
plt.suptitle('delta boxplot')
plt.savefig('../fig/delta_box.png')

# check summary statistics
d_r_long.groupby("rating_n")["r"].mean() # similar
d_r_long.groupby("rating_n")["r"].median() # ...

# more advanced modeling needed... 
# perhaps it is because I am averaging it out over contexts
# but, that should actually give less power (i.e. including random effects)

# quick t-test (is significant, but I doubt it holds with random effs.)
from scipy import stats
d_r1 = d_r_long[d_r_long["rating_n"] == 1]["r"].values
d_r2 = d_r_long[d_r_long["rating_n"] == 2]["r"].values
stats.ttest_ind(d_r1, d_r2)

# https://github.com/RainCloudPlots/RainCloudPlots

# rating vs. fluency # 
d_r_long.head(5)
d_r1_long = d_r_long[d_r_long["rating_n"] == 1]
sns.lmplot(
    d_r1_long,
    x = "rt",
    y = "r",
    hue = "condition" 
)
plt.suptitle('rating 1 vs. fluency')
plt.savefig("../fig/rating_fluency.png")