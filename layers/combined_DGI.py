import torch
import torch.nn as nn
from dgi import DGI
import numpy as np
from layers.gcn import GCN


# noinspection PyCallingNonCallable
class combinedDGI(nn.Module):
    def __init__(self, n_h, n_h2, n_motif, activation):
        super(combinedDGI, self).__init__()
        self.dgi_list = nn.ModuleList([DGI(n_h, n_h2, activation) for i in range(n_motif)])
        self.linear = nn.Linear(n_h, n_h2)
        self.count = n_motif
        self.n_h2 = n_h2
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x1, motif_batch, shuf_fts):
        logit_array = []
        x_array = []
        for i in range(self.count):
            x, logit = self.dgi_list[i](x1, shuf_fts, torch.LongTensor(motif_batch[i]).cuda())
            x_array.append(x)
            logit_array.append(logit)

        return x_array, logit_array

    def embed(self, x1):
        x_array = []
        for i in range(self.count):
            x_array.append(self.dgi_list[i].embed(x1))
        return x_array
