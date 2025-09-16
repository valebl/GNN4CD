import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv
import numpy as np
import torch

from typing import Optional
from torch import Tensor

from torch_geometric.nn.inits import ones, zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter


###########################################################
### The GNN4CD_model architecture is used for both the  ###
### Classifier and Regressor components of the Emulator ###
###########################################################

class GNN4CD_model(nn.Module):
    
    def __init__(self, encoding_dim=128, seq_l=25, h_in=5*5, h_hid=5*5, n_layers=2, high_in=6+1, low2high_out=64, high_out=64):
        super(GNN4CD_model, self).__init__()

        # input shape (N,L,Hin)
        self.rnn = nn.Sequential(
            nn.GRU(h_in, h_hid, n_layers, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(h_in*seq_l, encoding_dim),
            nn.ReLU()
        )

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv((encoding_dim, high_in), out_channels=low2high_out, aggr='mean'), 'x, edge_index -> x')
            ])
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=1, dropout=0.0, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )

    def forward(self, data, inference=False):
        encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
        encod_rnn = encod_rnn.flatten(start_dim=1)
        encod_rnn = self.dense(encod_rnn)
        encod_low2high  = self.downscaler((encod_rnn, data.x_dict['high']), data["low", "to", "high"].edge_index)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        x_high = self.predictor(encod_high)

        if inference:
            data['high'].x_high = x_high
            data._slice_dict['high']['x_high'] = data._slice_dict['high']['y']
            data._inc_dict['high']['x_high'] = data._inc_dict['high']['y']
            data = data.to_data_list()
            x_high = [data_i['high'].x_high for data_i in data]

        return x_high