import torch
import torch.nn as nn
import numpy as np

name     = 'hhnet'
features = ['hh_mass', 'klambda', 'ct', 'ctt', 'cggh', 'cgghh']
target   = 'target'
#play with the nodes
nodes    = 20
noutputs =  1

#reduce from 5 to something lower
model = nn.Sequential(nn.Linear(len(features), nodes), nn.SiLU(),
                      nn.Linear(nodes, nodes), nn.SiLU(),
                      nn.Linear(nodes, nodes), nn.SiLU(),
                      nn.Linear(nodes, nodes), nn.SiLU(),
                      nn.Linear(nodes, nodes), nn.SiLU(),
                      nn.Linear(nodes, noutputs), nn.Sigmoid()
                     )
