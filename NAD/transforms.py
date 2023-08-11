import copy

import torch
from torch_geometric.utils.dropout import dropout_adj, dropout_edge
from torch_geometric.transforms import Compose


class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, seed=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p
        self.seed = seed

    def __call__(self, data):
        # Set the seed for reproducibility
        if self.seed:
            torch.manual_seed(self.seed)
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class DropEdges:
    r"""Drops edges with probability p."""
    def __init__(self, p, seed=None, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected
        self.seed = seed

    def __call__(self, data):
        # Set the seed for reproducibility
        if self.seed:
            torch.manual_seed(self.seed)
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        edge_index, edge_attr = dropout_edge(edge_index, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)


def get_graph_drop_transform(drop_edge_p, drop_feat_p, seed=None):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p,seed=seed))

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p, seed=seed))
    return Compose(transforms)

