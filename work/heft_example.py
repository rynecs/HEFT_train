#!/usr/bin/env python
# coding: utf-8

# ## Higgs Effective Field Theory: Training
# > Created: Jun 12, 2024 Ryne Starnes and Harrison B. Prosper<br>
# 
# ### Introduction
# 
# In this notebook, we model the HEFT di-Higgs cross section[1] (per 15 GeV in the di-Higgs mass, $m_{hh}$) in which the 15 GeV bin size is mapped to a dimensionless value of 0.01. The HEFT parameter space is defined by the 5 parameters $\theta = \kappa_\Lambda ( \equiv c_{hhh}), c_{t}, c_{tt}, c_{ggh}, c_{gghh}$. The goal of this notebook is to approximate the probability density $p(x | \theta)$, where $x = m_{hh}$.
# If on command line, run in ipython to prevent any errors

# In[ ]:


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
import dnnutil as dn

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


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


# ### Load training data

# In[ ]:


datafile = '../work/nn_train_data.csv'

print('loading %s' % datafile)
df  = pd.read_csv(datafile)
print('number of rows: %d\n' % len(df))

df[:5]


# In[ ]:


# mean of dihiggs mass
mean = df['hh_mass'].mean()

# divide dihiggs masses by twice the mean
df['hh_mass'] = df['hh_mass']/(2*mean)

df[:5]


# ### Train, validation, and test sets
# There is some confusion in terminology regarding validation and test samples (or sets). We shall adhere to the defintions given here https://machinelearningmastery.com/difference-test-validation-datasets/):
#    
#   * __Training Dataset__: The sample of data used to fit the model.
#   * __Validation Dataset__: The sample of data used to decide 1) whether the fit is reasonable (e.g., the model has not been overfitted), 2) decide which of several models is the best and 3) tune model hyperparameters.
#   * __Test Dataset__: The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.
# 
# The validation set will be some small fraction of the training set and can be used, for example, to decide when to stop the training.

# In[ ]:


# Fraction of the data assigned as test data and validation
ntrain    = 390000                # training sample size
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


# This following bit looks much better in a jupyter notebook

# ### Empirical risk (that is, average loss)
# 
# The empirical risk, which is the **objective function** we shall minimize, is defined by
# 
# \begin{align}
# R_M(\omega) & = \frac{1}{M} \sum_{m=1}^{M} L(t_m, f_m),
# \end{align}
# 
# where 
# 
# \begin{align*}
#     f_m & \equiv f(\boldsymbol{x}_m, \omega)
# \end{align*}
# is the machine learning model with parameters $\omega$ to be determined by minimizing $R_M$. 
# The quantity $X =  x, \theta$ are the inputs to the model and the target $t = 1$ for $X \sim p(x, \theta) = p(x | \theta) \pi(\theta)$ and $t = 0$ for $X \sim f(x) \pi(\theta)$, where $\pi(\theta)$, the prior, is the same for both samples.

# ### Define neural network model

# In[ ]:


get_ipython().run_cell_magic('writefile', 'hhnet.py', "import torch\nimport torch.nn as nn\nimport numpy as np\n\nname     = 'hhnet'\nfeatures = ['hh_mass', 'klambda', 'ct', 'ctt', 'cggh', 'cgghh']\ntarget   = 'target'\n#play with the nodes\nnodes    = 20\nnoutputs =  1\n\n#reduce from 5 to something lower\nmodel = nn.Sequential(nn.Linear(len(features), nodes), nn.SiLU(),\n                      nn.Linear(nodes, nodes), nn.SiLU(),\n                      nn.Linear(nodes, nodes), nn.SiLU(),\n                      nn.Linear(nodes, nodes), nn.SiLU(),\n                      nn.Linear(nodes, nodes), nn.SiLU(),\n                      nn.Linear(nodes, noutputs), nn.Sigmoid()\n                     )\n")


# In[ ]:


import hhnet as NN
importlib.reload(NN)

name     = NN.name
model    = NN.model
features = NN.features
target   = NN.target

modelfile  = '%s.dict' % NN.name
print(name)
print(model)
print('number of parameters: %d\n' % dn.number_of_parameters(model))


# ### Train!

# In[ ]:


traces = ([], [], [])

traces_step   = 100
#play w batch size and lr
n_batch       =  64
n_iterations  = 500000
early_stopping=  50000
learning_rate = 1.e-3


# In[ ]:


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


# In[ ]:


Load_lowest_loss_model = True

if Load_lowest_loss_model:
    print('load lowest loss model dictionary: %s' % modelfile)
    modeldict = torch.load(modelfile)
    model = NN.HNet()
    model.load_state_dict(modeldict)


# In[ ]:




