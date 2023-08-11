import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torch_geometric.nn import GCNConv, BatchNorm, LayerNorm, Sequential, HypergraphConv, MLP
from utils import *

class BOURNE(nn.Module):
    r"""
    OurModel architecture for comprehensive anomaly detection task.

    Args:
        graph_encoder (torch.nn.Module): Encoder network used as online network
        hypergraph_encoder (torch.nn.Module): Encoder network used as target network
        predictor (torch.nn.Module): Predictor network used as projection from online network.
    """
    def __init__(self, encoder_g, encoder_h, predictor, subgraph_size, readout='wei'):
        super().__init__()
        # online network
        self.online_encoder = encoder_h
        self.predictor = predictor

        # target network
        self.target_encoder = encoder_g

        self.disc1 = Discriminator(encoder_h.out_dim)
        self.disc2 = Discriminator(encoder_g.out_dim)

        self.subgraph_size = subgraph_size

        # reinitialize weights
        self.target_encoder.reset_parameters()

        # readout func
        if readout == 'max':
            self.readout = MaxReadout()
        elif readout == 'avg':
            self.readout = AvgReadout()
        elif readout == 'wei':
            self.readout = MyReadout()
        else:
            raise ValueError('Readout type not defined.')
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)
        for param_q, param_k in zip(self.disc1.parameters(), self.disc2.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, data1, data2):
        node_emb_1 = self.online_encoder(data1.x, data1.edge_index)
        node_emb_1 = self.predictor(node_emb_1)
        h_n_1 = node_emb_1[-data1.batch_size:, :]
        h_sub_1 = self.readout(node_emb_1[:-data1.batch_size, :], data1.batch, 'mean')

        with torch.no_grad():
            node_emb_2 = self.target_encoder(data2.x, data2.edge_index)
            h_n_2 = node_emb_2[-data2.batch_size:, :]
            h_sub_2 = self.readout(node_emb_2[:-data2.batch_size, :], data2.batch, 'mean')

        #h_n_1, node_emb_2[:data1.batch_size, :], h_sub_2
        return h_n_1, node_emb_2[:data1.batch_size, :], h_sub_2


    def inference(self, data):
        node_emb_1 = self.online_encoder(data.x, data.edge_index)
        node_emb_1 = self.predictor(node_emb_1)
        h_n_1 = node_emb_1[-data.batch_size:, :]
        h_sub_1 = self.readout(node_emb_1[:-data.batch_size, :], data.batch, 'mean')


        node_emb_2 = self.target_encoder(data.x, data.edge_index)
        h_n_2 = node_emb_2[-data.batch_size:, :]
        h_sub_2 = self.readout(node_emb_2[:-data.batch_size, :], data.batch, 'mean')

        # dist_1 = 1 - cosine_similarity(node_emb_1[:data.batch_size, :],
        #                                h_n_2,
        #                                dim=-1).unsqueeze(-1)
        # dist_2 = 1 - cosine_similarity(node_emb_2[:data.batch_size, :],
        #                                h_n_1,
        #                                dim=-1).unsqueeze(-1)
        # dist_3 = 1 - cosine_similarity(node_emb_1[:data.batch_size, :],
        #                                h_sub_2,
        #                                dim=-1).unsqueeze(-1)
        # dist_4 = 1 - cosine_similarity(node_emb_2[:data.batch_size, :],
        #                                h_sub_1,
        #                                dim=-1).unsqueeze(-1)
        dist_1 = 1 - cosine_similarity(h_n_1,
                                       node_emb_2[:data.batch_size, :],
                                       dim=-1).unsqueeze(-1)
        dist_2 = 1 - cosine_similarity(h_n_1,
                                       h_sub_2,
                                       dim=-1).unsqueeze(-1)
        dist_3 = 1 - cosine_similarity(h_n_2,
                                       node_emb_1[:data.batch_size, :],
                                       dim=-1).unsqueeze(-1)
        dist_4 = 1 - cosine_similarity(h_n_2,
                                       h_sub_1,
                                       dim=-1).unsqueeze(-1)


        return dist_1, dist_2, dist_3, dist_4

class BOURNE_Edge(nn.Module):
    r"""
    OurModel architecture for comprehensive anomaly detection task.

    Args:
        graph_encoder (torch.nn.Module): Encoder network used as online network
        hypergraph_encoder (torch.nn.Module): Encoder network used as target network
        predictor (torch.nn.Module): Predictor network used as projection from online network.
    """
    # switch encoder_g and encoder_h
    def __init__(self, encoder_g, encoder_h, predictor, subgraph_size, readout='wei'):
        super().__init__()
        # online network
        self.online_encoder = encoder_h
        self.predictor = predictor

        # target network
        self.target_encoder = encoder_g

        self.disc1 = Discriminator(encoder_h.out_dim)
        self.disc2 = Discriminator(encoder_g.out_dim)

        self.subgraph_size = subgraph_size

        # reinitialize weights
        self.target_encoder.reset_parameters()

        # readout func
        if readout == 'max':
            self.readout = MaxReadout()
        elif readout == 'avg':
            self.readout = AvgReadout()
        elif readout == 'wei':
            self.readout = MyReadout()
        else:
            raise ValueError('Readout type not defined.')
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)
        for param_q, param_k in zip(self.disc1.parameters(), self.disc2.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, data1, data2):
        # # forward online Hyper GNN network
        edge_emb_1 = self.online_encoder(data1.edge_fea, data1.hyper_edge_index)

        # prediction
        edge_emb_1 = self.predictor(edge_emb_1)
        edge_emb_1 = edge_emb_1[data1.hyper_edge_index[0]]

        node_u = torch.zeros((data1.n_id.size(0) + data1.batch_size, edge_emb_1.size(-1)),
                                dtype=torch.float, device=data1.x.device)
        hyper_index = torch.broadcast_to(torch.unsqueeze(data1.hyper_edge_index[1],1),
                                         (edge_emb_1.size(0), edge_emb_1.size(1)))
        node_emb_1 = node_u.scatter_reduce_(0, hyper_index, edge_emb_1, reduce='mean')


        h_n_1 = node_emb_1[-data1.batch_size:, :]
        h_n_1_e = edge_emb_1[-data1.target_edge.size(0):, :]
        h_sub_1 = self.readout(node_emb_1[:-data1.batch_size, :], data1.batch, 'mean')
        # h_sub_1 = h_sub_1[data1.node_index]

        # forward target GNN network
        with torch.no_grad():
            node_emb_2 = self.target_encoder(data2.x, data2.edge_index)
            # h_n_2 = node_emb_2[-data2.batch_size:, :]
            h_n_2 = node_emb_2[:data2.batch_size, :]
            h_n_2_e = h_n_2[data1.node_index]
            h_sub_2 = self.readout(node_emb_2[:-data2.batch_size, :], data2.batch, 'mean')
            h_sub_2_e = h_sub_2[data1.node_index]


        #h_n_1, node_emb_2[:data2.batch_size, :], h_sub_2
        return h_n_1, node_emb_2[:data2.batch_size, :], h_sub_2


    def inference(self, data):
        edge_emb_1 = self.online_encoder(data.edge_fea, data.hyper_edge_index)

        edge_emb_1 = self.predictor(edge_emb_1)
        edge_emb_1 = edge_emb_1[data.hyper_edge_index[0]]

        node_u = torch.zeros((data.n_id.size(0) + data.batch_size, edge_emb_1.size(-1)),
                             dtype=torch.float, device=data.x.device)
        hyper_index = torch.broadcast_to(torch.unsqueeze(data.hyper_edge_index[1], 1),
                                         (edge_emb_1.size(0), edge_emb_1.size(1)))
        node_emb_1 = node_u.scatter_reduce_(0, hyper_index, edge_emb_1, reduce='mean')

        h_n_1_e = edge_emb_1[-data.target_edge.size(0):, :]
        h_sub_1 = self.readout(node_emb_1[:-data.batch_size, :], data.batch, 'mean')
        h_sub_1_e = h_sub_1[data.node_index]


        node_emb_2 = self.target_encoder(data.x, data.edge_index)
        h_n_2 = node_emb_2[-data.batch_size:, :]
        h_n_2_e = h_n_2[data.node_index]
        h_sub_2 = self.readout(node_emb_2[:-data.batch_size, :], data.batch, 'mean')
        h_sub_2_e = h_sub_2[data.node_index]


        dist_1 = 1 - cosine_similarity(h_n_1_e,
                                       edge_emb_1[data.target_edge, :],
                                       dim=-1).unsqueeze(-1)
        dist_2 = 1 - cosine_similarity(h_n_1_e,
                                       h_sub_2_e,
                                       dim=-1).unsqueeze(-1)
        dist_3 = 1 - cosine_similarity(h_n_2_e,
                                       edge_emb_1[data.target_edge, :],
                                       dim=-1).unsqueeze(-1)
        dist_4 = 1 - cosine_similarity(h_n_2_e,
                                       h_sub_1_e,
                                       dim=-1).unsqueeze(-1)


        return dist_1, dist_2, dist_3, dist_4

class BOURNE_Node(nn.Module):
    r"""
    OurModel architecture for comprehensive anomaly detection task.

    Args:
        graph_encoder (torch.nn.Module): Encoder network used as online network
        hypergraph_encoder (torch.nn.Module): Encoder network used as target network
        predictor (torch.nn.Module): Predictor network used as projection from online network.
    """
    def __init__(self, encoder_g, encoder_h, predictor, subgraph_size, readout='wei'):
        super().__init__()
        # online network
        self.online_encoder = encoder_g
        self.predictor = predictor

        # target network
        self.target_encoder = encoder_h

        self.disc1 = Discriminator(encoder_g.out_dim)
        self.disc2 = Discriminator(encoder_h.out_dim)

        self.subgraph_size = subgraph_size

        # reinitialize weights
        self.target_encoder.reset_parameters()

        # readout func
        if readout == 'max':
            self.readout = MaxReadout()
        elif readout == 'avg':
            self.readout = AvgReadout()
        elif readout == 'wei':
            self.readout = MyReadout()
        else:
            raise ValueError('Readout type not defined.')
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)
        for param_q, param_k in zip(self.disc1.parameters(), self.disc2.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, data1, data2):
        node_emb_1 = self.online_encoder(data1.x, data1.edge_index)
        h_n_1 = node_emb_1[-data1.batch_size:, :]
        h_sub_1 = self.readout(node_emb_1[:-data1.batch_size, :], data1.batch, 'mean')
        h_n_1_e = h_n_1[data1.node_index]
        h_sub_1_e = h_sub_1[data1.node_index]

        # forward target GNN network
        with torch.no_grad():
            edge_emb_2 = self.target_encoder(data2.edge_fea, data2.hyper_edge_index)
            edge_emb_2 = edge_emb_2[data2.hyper_edge_index[0]]

            node_u = torch.zeros((data2.n_id.size(0) + data2.batch_size, edge_emb_2.size(-1)),
                                 dtype=torch.float, device=data2.x.device)
            hyper_index = torch.broadcast_to(torch.unsqueeze(data2.hyper_edge_index[1], 1),
                                             (edge_emb_2.size(0), edge_emb_2.size(1)))
            node_emb_2 = node_u.scatter_reduce_(0, hyper_index, edge_emb_2, reduce='mean')

            h_n_2 = edge_emb_2[-data2.target_edge.size(0):, :]
            h_n_2_e = edge_emb_2[-data1.target_edge.size(0):, :]
            h_sub_2 = self.readout(node_emb_2[:-data2.batch_size, :], data2.batch, 'mean')
            h_sub_2_e = h_sub_2[data1.node_index]

        # h_n_1, node_emb_2[:data2.batch_size, :], h_sub_2

        return h_n_1, node_emb_2[:data2.batch_size, :], h_sub_2, \
               h_n_1_e, h_n_2_e, h_sub_2_e


    def inference(self, data):
        node_emb_1 = self.online_encoder(data.x, data.edge_index)
        h_n_1 = node_emb_1[-data.batch_size:, :]
        h_sub_1 = self.readout(node_emb_1[:-data.batch_size, :], data.batch, 'mean')
        h_sub_1_e = h_sub_1[data.node_index]

        edge_emb_2 = self.target_encoder(data.edge_fea, data.hyper_edge_index)
        edge_emb_2 = edge_emb_2[data.hyper_edge_index[0]]

        node_u = torch.zeros((data.n_id.size(0) + data.batch_size, edge_emb_2.size(-1)),
                             dtype=torch.float, device=data.x.device)
        hyper_index = torch.broadcast_to(torch.unsqueeze(data.hyper_edge_index[1], 1),
                                         (edge_emb_2.size(0), edge_emb_2.size(1)))
        node_emb_2 = node_u.scatter_reduce_(0, hyper_index, edge_emb_2, reduce='mean')
        h_n_2 = node_emb_2[-data.batch_size:, :]

        h_n_2_e = edge_emb_2[-data.target_edge.size(0):, :]
        h_sub_2 = self.readout(node_emb_2[:-data.batch_size, :], data.batch, 'mean')
        h_sub_2_e = h_sub_2[data.node_index]



        dist_1 = 1 - cosine_similarity(h_n_1,
                                       node_emb_2[:data.batch_size, :],
                                       dim=-1).unsqueeze(-1)
        dist_2 = 1 - cosine_similarity(h_n_1,
                                       h_sub_2,
                                       dim=-1).unsqueeze(-1)
        dist_3 = 1 - cosine_similarity(h_n_2,
                                       node_emb_1[:data.batch_size, :],
                                       dim=-1).unsqueeze(-1)
        dist_4 = 1 - cosine_similarity(h_n_2,
                                       h_sub_1,
                                       dim=-1).unsqueeze(-1)


        return dist_1, dist_2, dist_3, dist_4

class GCN(torch.nn.Module):
    # from bgrl
    def __init__(self, layer_sizes, act='prelu',
                 batch_norm=True, batchnorm_mm=0.99, layer_norm=False,
                 weight_standardization=False):
        super(GCN, self).__init__()
        self.num_layers = len(layer_sizes) - 1
        self.layers = torch.nn.ModuleList()

        assert batch_norm != layer_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.weight_standardization = weight_standardization
        self.out_dim = layer_sizes[-1]

        # Set the activation function
        if act == 'prelu':
            self.activation = torch.nn.PReLU()
        elif act == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif act == 'softmax':
            self.activation = torch.nn.Softmax()
        else:
            raise ValueError("Invalid activation function")

        for i in range(self.num_layers):
            in_channels = layer_sizes[i]
            out_channels = layer_sizes[i+1]
            self.layers.append(GCNConv(in_channels, out_channels))

            # Add batch normalization if specified
            if self.batch_norm:
                self.layers.append(BatchNorm(out_channels, momentum=batchnorm_mm))

            # Add layer normalization if specified
            if self.layer_norm:
                self.layers.append(LayerNorm(out_channels))


    def forward(self, x, edge_index):
        if self.weight_standardization:
            self.standardize_weights()

        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x, edge_index)
            x = self.layers[i + 1](x)
            x = self.activation(x)
        return x

    def reset_parameters(self):
        for module in self.layers:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.layers.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

class HGNN(torch.nn.Module):
    def __init__(self, layer_sizes, act='prelu',
                 batch_norm=False, batchnorm_mm=0.99, layer_norm=False,
                 weight_standardization=True):
        super(HGNN, self).__init__()
        self.num_layers = len(layer_sizes) - 1
        self.layers = torch.nn.ModuleList()
        self.out_dim = layer_sizes[-1]

        assert batch_norm != layer_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.weight_standardization = weight_standardization


        # Set the activation function
        if act == 'prelu':
            self.activation = torch.nn.PReLU()
        else:
            raise ValueError('Invalid activation function')

        for i in range(self.num_layers):
            in_channels = layer_sizes[i]
            out_channels = layer_sizes[i + 1]
            self.layers.append(HypergraphConv(in_channels, out_channels))

            # Add batch normalization if specified
            if self.batch_norm:
                self.layers.append(BatchNorm(out_channels, momentum=batchnorm_mm))

            # Add layer normalization if specified
            if self.layer_norm:
                self.layers.append(LayerNorm(out_channels))


    def forward(self, edge_fea, hyperedge_index, node_fea=None):
        if self.weight_standardization:
            self.standardize_weights()
        for i in range(0, len(self.layers), 2):
            edge_fea = self.layers[i](edge_fea, hyperedge_index, hyperedge_attr=node_fea)
            edge_fea = self.layers[i + 1](edge_fea)
            edge_fea = self.activation(edge_fea)

        return edge_fea

    def reset_parameters(self):
        for module in self.layers:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.layers.modules():
            if isinstance(m, HypergraphConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

class Discriminator(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self, n_h, negsamp_round=1):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        scs.append(self.f_k(h_pl, c))
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-1, :].unsqueeze(0), c_mi[:-1, :]), dim=0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits

class MaxReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values

class AvgReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class MyReadout(nn.Module):
    def __init__(self):
        super(MyReadout, self).__init__()

    def forward(self, seq, sub_match, reduce='mean'):
        out = torch.zeros((torch.unique(sub_match).size(0), seq.size(-1)),
                             dtype=torch.float, device=seq.device)
        sub_match = torch.broadcast_to(torch.unsqueeze(sub_match, 1),
                                       (sub_match.size(0), seq.size(1))).to(seq.device)
        out = out.scatter_reduce_(0, sub_match, seq, reduce=reduce)

        return out
def compute_representations(net, data, device):
    r"""Pre-computes the representations for the entire dataset.

    Returns:
        [torch.Tensor, torch.Tensor]: Representations and labels.
    """
    net.eval()
    reps = []
    labels = []

    # forward
    data = data.to(device)
    with torch.no_grad():
        reps.append(net(data.x, data.edge_index))
        labels.append(data.y)

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    reps = reps[:data.batch_size]
    labels = labels[:data.batch_size]

    return [reps, labels]
