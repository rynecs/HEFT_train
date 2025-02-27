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


datafile = f'../new_work&data/heft_gauss_traindata.csv'

print('loading %s' % datafile)
df  = pd.read_csv(datafile)
print('number of rows: %d\n' % len(df))

df['target'] = df.sigma

print(f'min(sigma):  {df.target.min():10.3f} pb, '\
      f'avg(sigma):  {df.target.mean():10.3f} pb,  max(sigma): {df.target.max():10.3f} pb\n')

df[:5]


spectra = pd.read_csv('../new_work&data/heft_gauss_spectra.csv')
len(spectra), spectra[:5]


# Fraction of the data assigned as test data and validation
ntrain    = 70000                # training sample size
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



import heftgaussnet as NN
importlib.reload(NN)

name     = NN.name
model    = NN.model
features = NN.features
target   = NN.target

modelfile  = '%s.dict' % NN.name
print(name)
print(model)
print('number of parameters: %d\n' % dn.number_of_parameters(model))

# check model
X = torch.Tensor(test_data[['mhh', 'klambda', 'CT', 'CTT', 'CGGH', 'CGGHH']].to_numpy())
print('input.size:  ', X.size())

Y = model(X)
print('output.size: ', Y.size())


def average_quadratic_loss_weighted(f, t, x=None):
    # f and t must be of the same shape
    w = torch.where(t != 0, 1/torch.abs(t), 1)
    return  torch.mean(w * (f - t)**2)


traces = ([], [], [])

traces_step   = 100
n_batch       = 150
n_epochs      = 8000
n_iterations  = n_epochs*(len(train_data) // n_batch)
early_stopping= 15000
learning_rate = 1.e-3

print(f'number of epochs:     {n_epochs:10d}')
print(f'number of iterations: {n_iterations:10d}')


av_loss = average_quadratic_loss_weighted

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


import dnnutil as dn
importlib.reload(dn)

class AveragedHEFTNet(dn.AveragedModel):
    
    def __init__(self, model, scale=0.01, size=25):
        
        super().__init__(model, scale, size)
        
    def coeffs(self, x):
        self.eval()
        
        y = self.models[0].coeffs(x)
        for m in self.models[1:]:
            y += m.coeffs(x)
        y /= len(self.models)
        return y


Load_lowest_loss_model = True
Average_model = False

if Load_lowest_loss_model:
    print('load lowest loss model dictionary: %s' % modelfile)
    modeldict = torch.load(modelfile)
    model = NN.model
    model.load_state_dict(modeldict)

if Average_model:
    print('create averaged model')
    model = AveragedHEFTNet(model, scale=0.0001, size=100)
    
# NN-approximated cross sections 
model.eval()
y = model(X).detach().numpy()

# POWHEG-predicted cross sections
t = test_data['target'].to_numpy()


def plot_results(y, t, 
                 xmin=0.0, xmax=2.0, 
                 ymin=0.0, ymax=2.0, 
                 ftsize=14, 
                 filename='fig_results.png'):

    # create an empty figure
    fig = plt.figure(figsize=(4, 4))
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows, ncols, index)
    
    ticks = np.linspace(xmin, xmax, 5)
    
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(ticks)
    ax.set_xlabel(r'$\sigma$ (predicted)', fontsize=ftsize)

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(ticks)
    ax.set_ylabel(r'$\sigma$ (NN)', fontsize=ftsize)
    
    ax.plot(y, t, 'b', marker='.', markersize=2, linestyle='')
    ax.plot([xmin, xmax], [ymin, ymax], linestyle='solid', color='red')

    ax.grid(True, which="both", linestyle='-')

    plt.savefig(filename)
    plt.show()

plot_results(y, t)


def get_spectra(model, df, row):

    # get column names
    columns= list(df.columns)
    params = columns[:5]
    bins   = columns[5:]
    
    xbins = len(bins)
    xmin  = 0
    xmax  = xbins/100
    
    # define mhh values
    x = np.linspace(xmin, xmax, xbins+1)
    x = (x[1:]+x[:-1])/2

    # get parameter names
    klambda, ct, ctt, cggh, cgghh = df[params].iloc[row]

    # create input data
    inputs = []
    for mhh in x:
        inputs.append([mhh, klambda, ct, ctt, cggh, cgghh])

    # get predicted cross section
    spectrum = df[bins].iloc[row].to_numpy() # predicted spectrum  

    # get approximated cross section
    inputs = torch.Tensor(np.array(inputs))
    
    model.eval()   
    y = model(inputs).detach().numpy()       # approximated spectrum 

    return spectrum.sum(), x, y, spectrum, klambda, ct, ctt, cggh, cgghh


def plot_spectra(data, 
                 ftsize=16, 
                 filename='fig_spectra_comparisons.pdf'):

    plt.rcParams.update({'font.size': 10})
    
    _, x, y, s, _,_,_,_,_ = data[0]
    xbins = len(x)
    xmin, ymin  = 0, 0
    xmax = xbins/100

    # create an empty figure
    fig = plt.figure(figsize=(10, 20))
    fig.tight_layout()

    # work out number of columns and number of plots
    ncols = 3
    nrows = len(data) // ncols
    ndata = nrows * ncols

    # loop over coefficients

    for i, (total_xsec, x, y, f, klambda, ct, ctt, cggh, cgghh) in enumerate(data):

        index = i+1
        ax  = fig.add_subplot(nrows, ncols, index)

        # setup x-axis
        ax.set_xlim(xmin, xmax)
        
        if i > (nrows-1) * ncols-1:
            ax.set_xlabel(r'$m_{hh}$', fontsize=ftsize)

        # setup y-axis
        ymax = 1.2 * f.max()
        ax.set_ylim(ymin, ymax)
        
        if i % ncols == 0:
            ax.set_ylabel(r'$\sigma$', fontsize=ftsize)

        # annotate plot
        xpos = xmin + 0.40 * (xmax-xmin)
        ypos = ymin + 0.80 * (ymax-ymin)
        ystep= (ymax-ymin)/6
        ax.text(xpos, ypos, r'$k_{lambda},c_{t},c_{tt},c_{ggh},c_{gghh}$'); ypos -= ystep
        ax.text(xpos, ypos, '%5.2f,%5.2f,%5.2f' % (ctt, cggh, cgghh)); ypos -= ystep
        
        xpos = xmin + 0.60 * (xmax-xmin)
        ypos -= ystep
        ax.text(xpos, ypos, r'$\sigma:$ %5.2f pb' % total_xsec)
        
        # predicted spectra
        ax.hist(x, bins=xbins, range=(xmin, xmax), weights=f, 
                    color='green', alpha=0.2)

        # NN-approximated spectra
        ax.plot(x, y, color='blue', linewidth=2)

    plt.savefig(filename)
    plt.show()

# ------------------------------------------------------------------
M = 39
data = []
for row in range(M):
    data.append( get_spectra(model, spectra, row) )

# sort in decreasing cross section
data.sort()

plot_spectra(data)