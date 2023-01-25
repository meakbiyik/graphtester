import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, global_add_pool


class GNN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units, dropout):
        super(GNN, self).__init__()

        dim = hidden_units
        self.dropout = dropout
        conv = GINConv

        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(conv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(num_features, num_classes))
        self.fcs.append(nn.Linear(dim, num_classes))

        for i in range(self.num_layers-1):
            self.convs.append(conv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, num_classes))
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        outs = [x]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x)
            
        out = None
        for i, x in enumerate(outs):
            x = global_add_pool(x, batch)
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x
        return F.log_softmax(out, dim=-1), 0
