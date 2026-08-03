#!/usr/bin/env python
# coding: utf-8

import pickle
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
import pickle

import sklearn.manifold
import sklearn.decomposition

np.set_printoptions(suppress=True)

df = pd.read_csv('../dftest.csv')
take = df.shape[0]//4

df = df.iloc[:take,:].reset_index(drop=True)
df['cif'] = [int(i[:7]) for i in df.name.values]
df = df[['cif']]

db = pd.read_csv('D:/latpar_update_0623/om_info.csv')
diccio = {k:v for v,k in enumerate(db.cif.values)}

idxs = df['cif'].map(diccio).values

means = list()
for size in ['0050','0100','0250','macro']:
    
    f = pickle.load(open(f"./latent_space_distribution_{size}.pkl",'rb'))
    means_t = f[f"{size}_means"]
    means += [means_t[idxs]]

xrd = np.concatenate(means, axis=0)
del means

xrd = np.reshape(xrd, (xrd.shape[0], 139, 5)) 
xrd = xrd.sum(axis=-1)

std_xrd = xrd.std(axis=0, keepdims=True)
mean_xrd = xrd.mean(axis=0, keepdims=True)

sxrd = (xrd - mean_xrd)/std_xrd


import sklearn.manifold


tsne_diccio = dict()
for perx in [5, 10, 20, 50, 80, 100, 150, 200, 300]:
    print('Starting t-SNE with perplexity', perx)
    tsne = sklearn.manifold.TSNE(perplexity=perx, max_iter=3000, random_state=3451, metric='cosine')
    tsne.fit(sxrd)
    
    tsne_diccio[perx] = tsne
    print('Done for perplexity', perx)
    
    plt.figure()
    plt.scatter(tsne.embedding_[:,0], tsne.embedding_[:,1], s=0.1)
    plt.savefig(f"flatten_tsne_perx_{perx}.png", dpi=300, transparent=True)
    plt.close('all')
    
with open('tsne_diccio_flatten_std.pkl','wb') as f:
    pickle.dump(tsne_diccio, f)

