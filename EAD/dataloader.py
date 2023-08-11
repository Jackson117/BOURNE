from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, Dict

import torch
from torch import Tensor

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader import NodeLoader, NeighborLoader
from torch_geometric.typing import InputNodes, EdgeType, NodeType
from torch_geometric.sampler import BaseSampler, \
    NodeSamplerInput, HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import SparseTensor
from utils import *


class RWR_Loader(NodeLoader):
    r"""A data loader that performs mini-batch sampling from node information, using
    the node_sampler of either neighbor sampler or random walk with restart.

    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        sampler: str = 'rwr',
        subgraph_size: int = 4,
        input_nodes: InputNodes = None,
        replace: bool = False,
        directed: bool = True,
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: bool = False,
        **kwargs,
    ):
        node_sampler = RWRSampler(
            data,
            subgraph_size= subgraph_size,
            replace=replace,
            directed=directed,
            disjoint=disjoint,
            is_sorted=is_sorted,
            share_memory=kwargs.get('num_workers', 0) > 0,
        )
        super().__init__(
            data=data,
            node_sampler=node_sampler,
            input_nodes=input_nodes,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )

class RWRSampler(BaseSampler):
    def __init__(self,
                 data: Union[Data, Tuple[FeatureStore, GraphStore]],
                 subgraph_size: int = 4,
                 replace: bool = False,
                 directed: bool = True,
                 disjoint: bool = False,
                 is_sorted: bool = False,
                 share_memory: bool = False
                 ):
        self.num_nodes = data.num_nodes

        self.data = data
        self.csr_adj = to_torch_csr_tensor(data.edge_index, data.edge_attr)
        self.subgraph_size = subgraph_size
        self.replace = replace
        self.directed = directed
        self.disjoint = disjoint
        self.is_sorted = is_sorted
        self.share_memory = share_memory

    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
        **kwargs,
    ) -> Union[HeteroSamplerOutput, SamplerOutput]:
        return self.node_sample(inputs, self._sample)

    def node_sample(
            self,
            inputs: NodeSamplerInput,
            sample_fn:Callable
    )->Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Performs sampling from a :class:`NodeSamplerInput`, leveraging a
        random walk with restart sampling function that accepts a seed as
        input. Returns the output of this sampling procedure."""
        if inputs.input_type is not None:  # Heterogeneous sampling:
            seed = {inputs.input_type: inputs.node}
            seed_time = None
            if inputs.time is not None:
                seed_time = {inputs.input_type: inputs.time}
        else:  # Homogeneous sampling:
            seed = inputs.node
            seed_time = inputs.time

        out = sample_fn(seed, seed_time)
        out.metadata = (inputs.input_id, inputs.time)

        return out

    def _sample(
            self,
            seed: Union[Tensor, Dict[NodeType, Tensor]],
            seed_time: Optional[Union[Tensor, Dict[NodeType, Tensor]]] = None,
            **kwargs,):
        r"""Implements random walk with restart sampling strategy.
        Output subgraphs have the shape of num_seed * subgraph_size"""

        out = generate_rwr_subgraph(seed, self.csr_adj, self.subgraph_size)
        node, row, col, edge, batch = out + (None, )


        return SamplerOutput(
            node=node,
            row=row,
            col=col,
            edge=edge,
            batch=batch,
        )
