import os, sys

import tkinter

# the standard module for tabular data
import pandas as pd

# the standard module for array manipulation
import numpy as np

# the standard modules for high-quality plots
import matplotlib as mp
mp.use('TkAgg')
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn

# split data into a training set and a test set
from sklearn.model_selection import train_test_split

# to reload modules
import importlib

# some simple dnn untilities
import dnnutil as dn

%matplotlib inline


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

## downaload data

datafile = '../work/freqtrain.csv'

print('loading %s' % datafile)
df  = pd.read_csv(datafile)
print('number of rows: %d\n' % len(df))

df[:5]

## sort data

# Fraction of the data assigned as test data and validation
ntrain    = 90000                # training sample size #90000
tfraction = (1-ntrain/len(df))/2 # test fraction
vfraction = tfraction            # validation fraction

# Split data into a part for training, validation, and testing
train_data, valid_data, test_data = dn.split_data(df, 
                                         test_fraction=tfraction, 
                                         validation_fraction=vfraction) 

print('train set size:        %6d' % train_data.shape[0])
print('validation set size:   %6d' % valid_data.shape[0])
print('test set size:         %6d' % test_data.shape[0])

train_data[:5]

## import model stuff

import hhfreq as NN
importlib.reload(NN)

name     = NN.name
model    = NN.model
features = NN.features
target   = NN.target


modelfile  = '%s.dict' % NN.name
print(name)
print(model)
print('number of parameters: %d\n' % dn.number_of_parameters(model))

## set model parameters

traces = ([], [], [])

traces_step   = 100
#play w batch size and lr
n_batch       =  128     #32
n_iterations  = 250000
early_stopping=  10000
learning_rate = 1.e-3   #1e-3

## run model

av_loss = dn.average_quadratic_loss
#try quadratic loss

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #adam

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
