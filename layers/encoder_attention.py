import torch
import torch.nn as nn


class Encoder_attention(nn.Module):
    def __init__(self, n_h):
        super(Encoder_attention, self).__init__()
        self.linear = nn.Linear(n_h, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Output: X """
        x1 = self.linear(x).squeeze()
        weights = self.softmax(x1).unsqueeze(2)

        x2 = torch.sum(torch.mul(x, weights), dim=1)  # x2.shape = (n_nodes, n_motifs, n_hidden)

        return x2, weights.squeeze().clone().detach()
