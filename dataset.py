import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData, Batch

import torch
import numpy as np
from torch_geometric.utils import degree

import torch_geometric.transforms as T
transform = T.AddLaplacianEigenvectorPE(k=2)

class Dataset_Graph(Dataset):

    def __init__(self, graph, targets, model_name, seq_l, **kwargs):
        self.graph = graph
        self.targets = targets
        self.model_name = model_name
        self.seq_l = seq_l
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._get_features = self.__get_features_tminus25_to_t_every_1h

    def __len__(self):
        return self.graph['low'].x.shape[1] # time dimension
        
    def _check_temporal_consistency(self):
        if self.targets is not None:
            assert (self.graph['low'].x.shape[1]) == self.targets.shape[1], "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self)
    
    def _get_high_nodes_degree(self, snapshot):
        node_degree = (degree(snapshot['high','within','high'].edge_index[0], snapshot['high'].num_nodes) / 8).unsqueeze(-1)
        return node_degree

    def __get_features_tminus25_to_t_every_1h(self, time_index: int):
        x_low = self.graph['low'].x[:,time_index-self.seq_l:time_index+1,:]     
        x_low = x_low.flatten(start_dim=2, end_dim=-1)                  
        return x_low
    
    def _get_target(self, time_index: int):
        return self.targets[:,time_index]

    def _get_train_mask(self, target: torch.tensor):
        return ~torch.isnan(target)
    
    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[:,time_index]
        return feature
    
    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features
    
    def set_t_offset(self, t_offset: int):
        self.t_offset = t_offset
    
    def __getitem__(self, time_index: int):
        x_low = self._get_features(time_index)
        y = self._get_target(time_index) if self.targets is not None else None
        train_mask = self._get_train_mask(y) if y is not None else None

        additional_features = self._get_additional_features(time_index)

        snapshot = HeteroData()

        for key, value in additional_features.items():
            if value.shape[0] == self.graph['high'].x.shape[0]:
                snapshot['high'][key] = value
            elif value.shape[0] == self.graph['low'].x.shape[0]:
                snapshot['high'][key] = value
       
        snapshot['high'].y = y
        snapshot['high'].train_mask = train_mask
        snapshot.num_nodes = self.graph.num_nodes
        snapshot['high'].num_nodes = self.graph['high'].num_nodes
        snapshot['low'].num_nodes = self.graph['low'].num_nodes
        snapshot.t = time_index - self.t_offset

        for edge_key in self.graph.edge_types: 
            if 'edge_index' in self.graph[edge_key]:  # Copy edge_index if it exists
                snapshot[edge_key].edge_index = self.graph[edge_key].edge_index
            if 'edge_attr' in self.graph[edge_key]:  # Copy edge_attr if it exists
                snapshot[edge_key].edge_attr = self.graph[edge_key].edge_attr

        snapshot['low'].x = x_low
        snapshot['high'].x = self.graph['high'].x

        snapshot['high'].lon = self.graph['high'].lon
        snapshot['high'].lat = self.graph['high'].lat
        snapshot['low'].lon = self.graph['low'].lon
        snapshot['low'].lat = self.graph['low'].lat

        return snapshot
        

class Iterable_Graph(object):

    def __init__(self, dataset_graph, shuffle, idxs_vector=None, t_offset=0):
        self.dataset_graph = dataset_graph
        self.dataset_graph.set_t_offset(t_offset)
        self.shuffle = shuffle
        self.idxs_vector = idxs_vector

    def __len__(self):
        return len(self.idxs_vector)

    def __next__(self):
        if self.prog_idx < self.idxs_vector.shape[0]:
            self.idx = self.sampling_vector[self.prog_idx].item()
            self.prog_idx = self.prog_idx + 1
            return self.idx
        else:
            self.prog_idx = 0
            self.idx = 0
            raise StopIteration

    def __iter__(self):
        self.prog_idx = 0
        self.idx = 0
        if self.idxs_vector is not None:
            if self.shuffle:
                rnd_idxs = torch.randperm(self.idxs_vector.shape[0])
                self.sampling_vector = self.idxs_vector[rnd_idxs].view(self.idxs_vector.size())
            else:
                self.sampling_vector = self.idxs_vector
        else:
            if self.shuffle:
                self.sampling_vector = torch.randperm(len(self)-24) + 24 # from 24 to len
            else:
                self.sampling_vector = torch.arange(24, len(self))
        return self


def custom_collate_fn_graph(batch_list):
    return Batch.from_data_list(batch_list)


