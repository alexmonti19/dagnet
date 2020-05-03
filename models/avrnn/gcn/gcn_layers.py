import math
import torch.nn as nn
import torch


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, input, adj):
        support = torch.mm(input, self.weight.cuda())
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias.cuda()
        else:
            return output

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)
        # self.bias.data.fill_(0.01)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
