import torch
import torch.nn as nn
import numpy as np

name     = 'heftgaussnet'
features = ['mhh', 'klambda', 'CT', 'CTT', 'CGGH', 'CGGHH']
target   = 'target'
nodes    = 15
noutputs =  1


#reduce from 5 to something lower
#added LayerNorm(nodes)(may not work)
model = nn.Sequential(nn.Linear(len(features), nodes), nn.SiLU(), nn.LayerNorm(nodes),
                      nn.Linear(nodes, nodes), nn.SiLU(), nn.LayerNorm(nodes),
                      nn.Linear(nodes, nodes), nn.SiLU(), nn.LayerNorm(nodes),
                      nn.Linear(nodes, nodes), nn.SiLU(), nn.LayerNorm(nodes),
                      nn.Linear(nodes, nodes), nn.SiLU(), nn.LayerNorm(nodes),
                      nn.Linear(nodes, noutputs)
                     )
