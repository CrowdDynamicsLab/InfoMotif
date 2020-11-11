import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, n_h):
        super(Attention, self).__init__()
        self.linear = nn.Linear(n_h * 2, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):

        curr_node = x[:, :, 0, :].unsqueeze(2).expand_as(x)
        stacked_x = torch.cat((curr_node, x), 3)

        x1 = self.linear(stacked_x).squeeze()
        weights = self.softmax(x1).unsqueeze(3)

        x2 = torch.sum(torch.mul(x, weights), dim=2)  # x2.shape = (n_nodes, n_motifs, n_hidden)
        x3 = torch.mean(x2, dim=1)  # x3.shape = (n_nodes, n_hidden)

        return x3
