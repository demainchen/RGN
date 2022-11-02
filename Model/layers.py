import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GAT(nn.Module):
    def __init__(self, in_feature, out_feature, dropout, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(in_feature, out_feature, dropout=dropout, concat=True) for _
                           in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(out_feature * nheads, out_feature, dropout=dropout, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, residual=None, concat=True, ):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.ReLU()
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):

        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GCN_GAT(nn.Module):
    def __init__(self, infeature, outfeature, lamda, alpha,
                 dropout, heads):
        super(GCN_GAT, self).__init__()

        self.lamda = lamda
        self.alpha = alpha

        self.gat = GAT(infeature, outfeature, dropout, heads)
        self.gcn = GraphConvolution(infeature, outfeature, residual=False)

        self.cat_lin=nn.Linear(infeature*2,outfeature)

    # layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.alpha, i + 1))
    def forward(self, x, adj, h0, l):
        input = x
        x1 = self.gat(x, adj)
        x2 = self.gcn(x, adj, h0, self.lamda, self.alpha, l)
        x=torch.cat((x1,x2),dim=1)
        x=self.cat_lin(x)
        return input + x


class deepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant, heads):
        super(deepGCN, self).__init__()
        self.convs = nn.ModuleList()

        self.ln=nn.LayerNorm(nhidden)
        for _ in range(nlayers):
            # self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant, residual=True))
            self.convs.append(
                GCN_GAT(infeature=nhidden, outfeature=nhidden, lamda=lamda, alpha=alpha, dropout=dropout, heads=heads))
            # self.convs.append(GraphAttentionLayer(nhidden,nhidden,dropout=0.3))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        self.lin1=nn.Linear(54,256)
        self.lin2 = nn.Linear(1024, 256)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def forward(self, x, adj,seq_emb):

        x=self.lin1(x)
        seq_emb=self.lin2(seq_emb)

        x=torch.cat((x,seq_emb),dim=1)

        # x=self.transformer_encoder(x).squeeze(dim=0)

        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(self.ln(con(layer_inner, adj, _layers[0], i + 1)))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner


