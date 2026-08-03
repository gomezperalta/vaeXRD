#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


df = pd.read_csv('../dftest.csv')
take = df.shape[0]//4

df = df.iloc[:take,:].reset_index(drop=True)
df['cif'] = [int(i[:7]) for i in df.name.values]
df = df[['cif']]

db = pd.read_csv('D:/latpar_update_0623/om_info.csv')
diccio = {k:v for v,k in enumerate(db.cif.values)}

idxs = df['cif'].map(diccio).values

xrd = list()
for size in ['0050','0100','0250','macro']:
    xrd += [np.load(f"D:/latpar_update_0623/xset_{size}.npy")[idxs]]
xrd = np.concatenate(xrd, axis=0)

xrd = np.reshape(xrd, (xrd.shape[0], 139, -1))
xrd = xrd.sum(axis=-1)

std_xrd = xrd.std(axis=0, keepdims=True)
mean_xrd = xrd.mean(axis=0, keepdims=True)

sxrd = (xrd - mean_xrd)/std_xrd

import sklearn.manifold

tsne_diccio = dict()
for ee in [5, 10, 20, 50, 80, 100]:
    print('Starting t-SNE with early_exaggeration', ee)
    tsne = sklearn.manifold.TSNE(perplexity=300, early_exaggeration = ee, max_iter=3000, random_state=3451, metric='cosine')
    tsne.fit(sxrd)
    
    tsne_diccio[ee] = tsne
    print('Done for early_exaggeration', ee)
    
    plt.figure()
    plt.scatter(tsne.embedding_[:,0], tsne.embedding_[:,1], s=0.1)
    plt.savefig(f"tsne_ee_{ee}.png", dpi=300, transparent=True)
    plt.close('all')
    
with open('tsne_diccio_300_ee.pkl','wb') as f:
    pickle.dump(tsne_diccio, f)

