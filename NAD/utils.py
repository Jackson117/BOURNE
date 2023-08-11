import numpy
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F

import scipy.io as sio
import random
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier

from torch_geometric.data import Data
from torch_geometric.utils import to_torch_csr_tensor, add_self_loops
from torch_geometric.datasets import Planetoid
from tqdm import tqdm


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def to_edge_index(adj):
    if isinstance(adj, torch.Tensor):
        row, col, value = adj.to_sparse_coo().indices()[0], adj.to_sparse_coo().indices()[1], \
                        adj.to_sparse_coo().values()

    elif isinstance(adj, sp.csr_matrix):
        row, col, value = adj.tocoo().row, adj.tocoo().col, \
                          adj.tocoo().data
        row, col, value = torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long), \
                          torch.tensor(value, dtype=torch.float)
    else:
        raise RuntimeError("adj has to be either torch.sparse_csr_matrix or scipy.sparse.csr_matrix.")
    if value is None:
        value = torch.ones(row.size(0), device=row.device)

    return torch.stack([row, col], dim=0), value


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def accuracy(preds, labels):
    # preds = torch.round(nn.Sigmoid()(logits))

    if labels.max() > 1:
        AUC = roc_auc_score(labels, preds, average='macro',
                        multi_class='ovr')
    else:
        AUC = roc_auc_score(labels, preds, average='macro')
    preds = np.where(preds >= 0.5, 1, 0)
    recall = recall_score(labels, preds, average='macro')
    macro_f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')

    return recall, macro_f1, AUC, acc, precision


def generate_rwr_subgraph(node_idx, adj, subgraph_size):
    reduced_size = subgraph_size - 1

    traces = random_walk_with_restart(
        node_idx,
        adj,
        restart_prob=1.0,
        num_steps=subgraph_size * 3
    )
    subv = []
    for i, trace in enumerate(traces):
        subv.append(torch.unique(torch.tensor(trace), sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = random_walk_with_restart(
                [node_idx[i]],
                adj,
                restart_prob=0.9,
                num_steps=subgraph_size * 5
            )
            subv[i] = torch.unique(torch.tensor(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= reduced_size) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)

        subv[i] = subv[i][:reduced_size]
        subv[i].append(node_idx[i].item())

    # nodes in subgraph
    node = torch.tensor(subv, dtype=torch.long).flatten()
    node = node.to(adj.device)

    edge_index, edge_attr = to_edge_index(adj)
    # edges in subgraph
    sub_edge_set = []
    edge = []
    for i, pair in enumerate(edge_index.T):
        u, v = pair[0], pair[1]
        if u in node and v in node:
            sub_edge_set.append([u, v])
            edge.append(i)
    sub_edge_index = torch.tensor(sub_edge_set).T
    row, col = sub_edge_index[0, :], sub_edge_index[1, :]
    edge = torch.tensor(edge, dtype=torch.long)

    return node, row, col, edge


def random_walk_with_restart(start_nodes, adj, restart_prob, num_steps, handles_deadend=True):
    """
    Computes the RWR scores for each node in the graph, starting from given start nodes.

    Parameters
    ----------
    start_nodes : Tensor
        Tensor of start nodes for the random walk.
    adj : csr sparse matrix
        Graph data in sparse matrix format.
    restart_prob : float
        Probability of restarting at each step.
    num_steps : int
        Maximum number of steps to perform.

    Returns
    -------
    list
        Node traces starting from the given start nodes.
    """

    n = adj.to_dense().shape[0]
    rwr_scores = torch.zeros(n, dtype=torch.float32, device=adj.device)
    old_rwr_scores = rwr_scores
    residuals = np.zeros(num_steps)

    traces = []
    for start_node in start_nodes:
        trace = [int(start_node)]
        rwr_scores[start_node] = 1.0
        for i in range(num_steps):
            if handles_deadend:
                rwr_scores = (1 - restart_prob) * (torch.mv(adj, old_rwr_scores))
                S = torch.sum(rwr_scores)
                rwr_scores = rwr_scores + (1 - S) * rwr_scores
            else:
                rwr_scores = (1 - restart_prob) * (torch.mv(adj, old_rwr_scores)) + (restart_prob * rwr_scores)

            residuals[i] = torch.linalg.vector_norm(rwr_scores - old_rwr_scores, ord=1)

            if residuals[i] <= 1e-9:
                break

            old_rwr_scores = rwr_scores
            trace.append(int(rwr_scores.argmax()))

        traces.append(trace)


    return traces


def dual_hypergraph_trans(edge_index, batch, add_loops=False):

    num_edge = edge_index.size(1)
    device = edge_index.device

    ### Transform edge list of the original graph to hyperedge list of the dual hypergraph
    edge_to_node_index = torch.arange(0,num_edge,1, device=device).repeat_interleave(2).view(1,-1)
    hyperedge_index = edge_index.T.reshape(1,-1)
    hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long()

    ### Transform batch of nodes to batch of edges
    if batch is not None:
        edge_batch = hyperedge_index[1,:].reshape(-1,2)[:,0]
        edge_batch = torch.index_select(batch, 0, edge_batch)
    else:
        edge_batch = None

    ### Add self-loops to each node in the dual hypergraph
    if add_loops:
        bincount =  hyperedge_index[1].bincount()
        mask = bincount[hyperedge_index[1]]!=1
        max_edge = hyperedge_index[1].max()
        loops = torch.cat([torch.arange(0,num_edge,1,device=device).view(1,-1),
                            torch.arange(max_edge+1,max_edge+num_edge+1,1,device=device).view(1,-1)],
                            dim=0)

        hyperedge_index = torch.cat([hyperedge_index[:,mask], loops], dim=1)

    return hyperedge_index, edge_batch

def permute_hypergraph(hyperedge_index, aug_ratio, add_e=True):
    r"""
    Treat the input hypergraph as a equivalent bipartite graph and add/drop edges
    of the bipartite graph as permutation.
    """
    device = hyperedge_index.device
    hyperedge_index = hyperedge_index.detach().cpu().numpy()
    node_num = int(hyperedge_index.shape[1] / 2)
    _, edge_num = hyperedge_index.shape
    hyperedge_num = np.unique(hyperedge_index[1]).shape[0]

    permute_num = int((edge_num - node_num) * aug_ratio)
    if add_e:
        # added edges in bipartite graph with randomly selected nodes and hyperedges
        idx_add_1 = np.random.choice(hyperedge_index[0], permute_num)
        idx_add_2 = np.random.choice(hyperedge_index[1], permute_num)
        idx_add = np.stack((idx_add_1, idx_add_2), axis=0)
    else:
        idx_add = None
    edge2remove_index = np.where(hyperedge_index[1] < hyperedge_num)[0]
    edge2keep_index = np.where(hyperedge_index[1] >= hyperedge_num)[0]
    edge_keep_index = np.random.choice(edge2remove_index, (edge_num - int(node_num * aug_ratio)) - permute_num, replace=False)
    edge_after_remove1 = hyperedge_index[:, edge_keep_index]
    edge_after_remove2 = hyperedge_index[:, edge2keep_index]

    if add_e:
        hyperedge_index = np.concatenate((edge_after_remove1, edge_after_remove2, idx_add), axis=1)
    else:
        # edge_index = edge_after_remove
        hyperedge_index = np.concatenate((edge_after_remove1, edge_after_remove2), axis=1)

    return torch.from_numpy(hyperedge_index).to(device)

def get_index(hyper_edge_index, n_id):
    hyper_index = hyper_edge_index[1,:]
    n_id_u = torch.unique(n_id)
    n_index = torch.from_numpy(np.arange(0, n_id_u.size(0), 1))
    dic = {n_id_u[i].item():n_index[i].item() for i in range(n_id_u.size(0))}
    hyper_index = torch.tensor([dic[hyper_index[i].item()] for i in range(hyper_index.size(0))],
                               device=hyper_edge_index.device)
    batch_index = torch.tensor([dic[n_id[i].item()] for i in range(n_id.size(0))],
                               device=n_id.device)

    return hyper_index, batch_index

def trans2hyper(b, aug_ratio):
    device = b.x.device
    # Dual hypergraph transformation
    hyper_edge_index, edge_batch = dual_hypergraph_trans(b.edge_index, batch=None)

    # Permute hypergraph
    # hyper_edge_index_p = permute_hypergraph(hyper_edge_index, aug_ratio, add_e=True)
    hyper_edge_index_p = hyper_edge_index

    # Add self-loop for target edges
    target_pairs = []
    cur_num, cur_num_e, add_num = b.n_id.size(0), hyper_edge_index.size(1) / 2, b.batch_size
    for e, n in hyper_edge_index_p.T.detach().cpu().numpy():
        if n < add_num:
            target_pairs.append([e, n + cur_num])

    target_edge, indices = np.unique(np.array(target_pairs).T[0], return_index=True)
    edge_map = {t : i + cur_num_e for i, t in enumerate(target_edge)}

    edge_ten = torch.unsqueeze(torch.tensor([edge_map[t[0]] for t in target_pairs], dtype=torch.long, device=device), dim=0)
    node_ten = torch.unsqueeze(torch.tensor(target_pairs, dtype=torch.long, device=device).T[1], dim=0)
    node_index = torch.squeeze(node_ten)[indices] - cur_num

    add_hyperedge = torch.cat((edge_ten, node_ten), dim=0)

    hyper_edge_index = torch.cat((hyper_edge_index_p, add_hyperedge), dim=1)

    # Compute edge_fea
    edge_fea = (b.x[b.edge_index[0, :]] + b.x[b.edge_index[1, :]]) / 2
    # edge_fea = b.x[b.edge_index[0, :]]
    edge_fea = torch.cat((edge_fea, edge_fea[target_edge]), dim=0)
    edge_fea[target_edge] = 0.

    b.edge_fea, b.hyper_edge_index = edge_fea, hyper_edge_index
    b.target_edge = torch.tensor(target_edge, device=device)
    b.node_index = node_index

    return b


def add_zero(data):
    device = data.x.device
    x, edge_index = data.x, data.edge_index

    zero_feas = torch.zeros((data.batch_size, x.size(1)), dtype=torch.float, device=device)
    x = torch.cat((zero_feas, x[data.batch_size:, :], x[:data.batch_size, :]), dim=0)

    cur_num, add_num = data.n_id.size(0), data.batch_size
    add_loop = torch.tensor(np.arange(cur_num, cur_num + add_num), dtype=torch.long, device=device).repeat(2,1)
    edge_index = torch.cat((edge_index, add_loop), dim=1)

    data.x, data.edge_index = x, edge_index

    return data

def insert_anomaly_edges(data, selected_nodes, k):
    """
    This function randomly inserts K anomaly edges for each source node in the selected
    node set,. The target node should not be a neighbor of the source node.
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index.t().tolist()

    ano_edge_index = []
    for i, pair in enumerate(edge_index):
        if pair[0] in selected_nodes or pair[1] in selected_nodes:
            ano_edge_index.append(i)
    ano_edge_index = np.random.choice(ano_edge_index, selected_nodes.size(0) * k, replace=False)

    edge_ano_label = torch.zeros(len(edge_index), dtype=torch.long, device=data.x.device)
    edge_ano_label[ano_edge_index] = 1

    data.edge_ano_label = edge_ano_label

    return data

def select_source_node(data):
    num_edges = data.edge_index.size(1)
    num_nodes = data.num_nodes
    num_ano = int(torch.count_nonzero(data.y))

    inserted_rate = 0.01
    inserted_num = int(num_edges * inserted_rate / 2) # k=2
    node_id = np.arange(num_nodes)
    if inserted_num <= num_ano:
        selected_node = np.random.choice(node_id[data.y == 1], inserted_num, replace=False)
    else:
        norm_node = np.random.choice(node_id[data.y == 0], inserted_num - num_ano, replace=False)
        selected_node = np.concatenate((norm_node, node_id[data.y==1]))

    return torch.tensor(selected_node, dtype=torch.long)

