import os, sys

# the standard module for tabular data
import pandas as pd

# the standard module for array manipulation
import numpy as np

# the standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn

# split data into a training set and a test set
from sklearn.model_selection import train_test_split

# to reload modules
import importlib

# some simple dnn untilities
import dnnutilf as dn

# update fonts
FONTSIZE = 12
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set usetex = False if LaTex is not 
# available on your system or if the 
# rendering is too slow
mp.rc('text', usetex=False)

# set a seed to ensure reproducibility
seed = 42
rnd  = np.random.RandomState(seed)




datafile = '../freq/freqtrain.csv'

df  = pd.read_csv(datafile)

# mean of dihiggs mass
mean = df['mhh'].mean()

# divide dihiggs masses by twice the mean
df['mhh'] = df['mhh']/(2*mean)
df['freq'] = df['freq']/10

# mean of dihiggs mass
mean = df['mhh'].mean()

# divide dihiggs masses by twice the mean
df['mhh'] = df['mhh']/(2*mean)
df['freq'] = df['freq']/10

# Fraction of the data assigned as test data and validation
ntrain    = 190000                # training sample size
tfraction = (1-ntrain/len(df))/2 # test fraction
vfraction = tfraction            # validation fraction

# Split data into a part for training, validation, and testing
train_data, valid_data, test_data = dn.split_data(df, 
                                         test_fraction=tfraction, 
                                         validation_fraction=vfraction) 

import hhfreq as NN
importlib.reload(NN)

name     = NN.name
model    = NN.model
features = NN.features
target   = NN.target


modelfile  = '%s.dict' % NN.name

traces = ([], [], [])

traces_step   = 100
#play w batch size and lr
n_batch       =  64
n_iterations  = 500000
early_stopping=  500
learning_rate = 1.e-3

av_loss = dn.average_quadratic_loss
#try quadratic loss

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

traces = dn.train(model, optimizer, 
                  modelfile, early_stopping,
                  av_loss,
                  dn.get_batch, 
                  train_data, valid_data,
                  features, target,
                  n_batch, 
                  n_iterations,
                  traces,
                  step=traces_step)

dn.plot_average_loss(traces)

Load_lowest_loss_model = True

if Load_lowest_loss_model:
    print('load lowest loss model dictionary: %s' % modelfile)
    modeldict = torch.load(modelfile)
    model = NN.model
    model.load_state_dict(modeldict)
