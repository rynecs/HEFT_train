import torch
import torch.nn as nn
import numpy as np

name     = 'hhfreq'
features = ['mhh', 'klambda', 'ct', 'ctt', 'cggh', 'cgghh']
target   = 'freq' #changed from target to freq
#play with the nodes
nodes    = 20
noutputs =  1


#reduce from 5 to something lower
#added layernorm(may not work)
model = nn.Sequential(nn.Linear(len(features), nodes), nn.SiLU(), nn.LayerNorm(nodes),
                      nn.Linear(nodes, nodes), nn.SiLU(), nn.LayerNorm(nodes),
                      nn.Linear(nodes, nodes), nn.SiLU(), nn.LayerNorm(nodes),
                      nn.Linear(nodes, nodes), nn.SiLU(), nn.LayerNorm(nodes),
                      nn.Linear(nodes, nodes), nn.SiLU(), nn.LayerNorm(nodes),
                      nn.Linear(nodes, noutputs)
                     )