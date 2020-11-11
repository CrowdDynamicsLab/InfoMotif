import torch
import torch.nn as nn
import numpy as np
from gcn import GCN
from readout import AvgReadout
from discriminator import Discriminator
from attention import Attention


# noinspection PyCallingNonCallable
class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.dense = nn.Linear(n_h, n_h)
        self.read = AvgReadout()
        self.attention = Attention(n_h)
        self.sigm = nn.Sigmoid()
        self.gcn = GCN(n_in, n_h, activation)
        self.disc = Discriminator(n_h)
        self.act = nn.PReLU()

    def forward(self, seq1, seq2, motif_batch):

        positive_seq = self.sigm(self.dense(seq1))
        positive_seq = torch.mul(positive_seq, seq1)
        negative_seq = self.sigm(self.dense(seq2))
        negative_seq = torch.mul(negative_seq, seq2)

        h_1 = positive_seq.squeeze()[motif_batch]  # should be of shape numberOfNodesInBatch * number of instances * 3 * n_hidden
        h_2 = negative_seq.squeeze()[motif_batch]

        positive = self.attention(h_1)
        negative = self.attention(h_2)

        c = self.read(positive, None)
        c = self.sigm(c)

        ret = self.disc(torch.unsqueeze(c, 0), torch.unsqueeze(positive, 0), torch.unsqueeze(negative, 0), None, None)
        return positive_seq, ret

    # Detach the return variables
    def embed(self, seq):

        h_1 = self.sigm(self.dense(seq))
        h_1 = torch.mul(h_1, seq)
        return h_1
