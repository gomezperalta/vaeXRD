#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import copy
import pickle

np.set_printoptions(suppress=True)

gpu_number = 1 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

n_hidden = 139*5

vae = tf.keras.models.load_model('./vae.h5')
vae.trainable = False

directorio_coleccion = '/home/bokhimi/iG_2023/latpar_update_0623'

for size in ['0050','0100','0250','macro']:

    diccio = dict()
    xrd = np.load(f"{directorio_coleccion}/xset_{size}.npy")
        
    _, means, stds = vae.predict(xrd, batch_size=128)
    diccio[f"{size}_means"] = means
    diccio[f"{size}_stds"] = stds

    with open(f"latent_space_distribution_{size}.pkl",'wb') as f:
        pickle.dump(diccio, f)

    print('Done for', size)

