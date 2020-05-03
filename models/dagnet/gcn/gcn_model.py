import torch
import torch.nn as nn
from .gcn_layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nin, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.bn2 = nn.BatchNorm1d(nout)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.bn1(x)
        x = self.gc2(x, adj)
        x = self.bn2(x)
        return torch.tanh(x)