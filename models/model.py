import torch
import torch.nn as nn
from layers.gcn import GCN
from layers.combined_DGI import combinedDGI
from layers.encoder_attention import Encoder_attention


class motif_emb(nn.Module):
    def __init__(self, n_in, n_h, n_h2, n_motif, n_class, activation="prelu", dropout=0.5):
        super(motif_emb, self).__init__()
        self.combined = combinedDGI(n_h, n_h2, n_motif, activation)
        self.linear = nn.Linear(n_h, n_class)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)
        self.attention = Encoder_attention(n_h2)
        self.gcn = GCN(n_in, n_h, activation)
        self.act = nn.PReLU()

    def forward(self, x, adj, motif_batch, shuf_fts, sparse=True):
        x1 = self.dropout(self.act(self.gcn(x, adj, sparse)))
        shuf_fts = self.dropout(self.act(self.gcn(shuf_fts, adj, sparse)))

        x2, logits = self.combined(x1, motif_batch, shuf_fts)
        if len(motif_batch) > 1:
            x3 = torch.cat(x2, dim=0)
            x4, weights = self.attention(x3.permute(1, 0, 2))
        else:
            x4 = x2[0]
            weights = torch.ones(x.shape[1], 1).cuda()
        weights = torch.cat(2 * [weights]).permute(1, 0)
        predicts = self.softmax(self.linear(x4.squeeze()))
        return logits, predicts, weights

    def predict(self, x, adj, index, sparse=True):
        index = torch.LongTensor(index)
        x1 = self.act(self.gcn(x, adj, sparse))
        x2 = self.combined.embed(x1)

        x3 = torch.cat(x2, dim=0)
        if len(x2) > 1:
            x4, weights = self.attention(x3.permute(1, 0, 2))
        else:
            x4 = x2[0]

        x2 = self.softmax(self.linear(x4.squeeze()))
        return x2[index].squeeze()
