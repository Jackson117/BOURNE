import copy

import numpy as np
import torch
from torch.optim import AdamW
from torch_geometric.datasets import AttributedGraphDataset
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm

from pygod.metric import eval_roc_auc, eval_precision_at_k, eval_recall_at_k
from memory_profiler import profile
import time

from dataloader import *
from utils import *
from model import *
from scheduler import *
from transforms import *

parser = argparse.ArgumentParser(description='BOURNE')
parser.add_argument('--cudaID', type=int, default=0)
parser.add_argument('--model_seed', type=int)
parser.add_argument('--dataset', type=str, default='cora',help='Choose a dataset from \'ACM\', \'Flickr\',\'cora\','
                                                               '\'citeseer\', \'pubmed\' and \'BlogCatalog\'.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--eval_epochs', type=int, default=1)
parser.add_argument('--eval_rounds', type=int, default=180)
parser.add_argument('--layer_sizes', nargs='+', type=int, required=True, default=256)
parser.add_argument('--predictor_hidden_size', type=int, default=512)
parser.add_argument('--num_neigh',type=int, default=4)
parser.add_argument('--no_edge', action='store_false')
parser.add_argument('--sample', type=str, default='nei',
                    help='Choose a sampling strategy from \'rwr\' (random walk with restart)'
                         'or \'nei\' (neighbor sampling)')
parser.add_argument('--aug_ratio', type=float, default=0.1)
parser.add_argument('--readout', type=str, default='wei')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_warmup_epochs', type=int, default=500)
parser.add_argument('--mm', type=float, default=0.99)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--alpha', type=float, default=0.7)
parser.add_argument('--beta', type=float, default=0.3)

args = parser.parse_args()

if __name__ == '__main__':
    # Set device
    device = torch.device(args.cudaID) if torch.cuda.is_available() else torch.device('cpu')

    # Set random seed
    if args.model_seed is not None:
        set_random_seeds(args.model_seed)

    # Load data
    print("Dataset is {}".format(args.dataset))
    dataset = AttributedGraphDataset('./dataset', name=args.dataset)
    data = dataset[0]
    data.x = data.x.to_dense()


    data = data.to(device)


    # Build networks
    layer_sizes = args.layer_sizes.insert(0, data.x.size(1))

    graph_encoder = GCN(args.layer_sizes, batch_norm=True)
    hypergraph_encoder = HGNN(args.layer_sizes, batch_norm=True)
    graph_encoder_2 = copy.deepcopy(graph_encoder)
    predictor = MLP_Predictor(args.layer_sizes[-1], args.layer_sizes[-1],
                              hidden_size=args.predictor_hidden_size)

    # Define model and dataloader
    if args.sample == 'nei' and args.no_edge ==True:
        model = BOURNE(
            graph_encoder,
            graph_encoder_2,
            predictor,
            subgraph_size=args.num_neigh,
            readout=args.readout).to(device)
        data_loader = NeighborLoader(
            data,
            num_neighbors=[args.num_neigh, int(args.num_neigh/2)],
            replace=True,
            disjoint=True,
            batch_size=args.batch_size)
    elif args.sample == 'nei' and args.no_edge == False:
        model = BOURNE_Node(
            graph_encoder,
            hypergraph_encoder,
            predictor,
            subgraph_size=args.num_neigh,
            readout=args.readout).to(device)
        data_loader = NeighborLoader(
            data,
            num_neighbors=[args.num_neigh, int(args.num_neigh / 2)],
            replace=True,
            disjoint=True,
            batch_size=args.batch_size)
    elif args.sample == 'rwr':
        model = BOURNE_Node(
            graph_encoder,
            hypergraph_encoder,
            predictor,
            subgraph_size=args.subgraph_size,
            readout=args.readout).to(device)
        data_loader = RWR_Loader(
            data,
            subgraph_size=args.subgraph_size,
            replace=True,
            disjoint=True,
            batch_size=args.batch_size)
    else:
        raise ValueError('args.sample not validated, plz choose from \'nei\' or \'rwr\'.')

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(args.lr, args.lr_warmup_epochs, args.epochs)
    mm_scheduler = CosineDecayScheduler(1 - args.mm, 0, args.epochs)

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=0.4, drop_feat_p=0.1) # 0.4, 0.1
    transform_2 = get_graph_drop_transform(drop_edge_p=0.1, drop_feat_p=0.4) # 0.1, 0.4

    # @profile(precision=4)
    def train(step, b1, b2):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for params in optimizer.param_groups:
            params['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward propagation
        optimizer.zero_grad()

        q1, y2, sub2 = model(b1, b2)
        q2, y1, sub1 = model(b2, b1)

        loss_q1y2 = 1 - cosine_similarity(q1, y2.detach(), dim=-1).mean()
        loss_q1sub2 = 1 - cosine_similarity(q1, sub2.detach(), dim=-1).mean()
        loss_q2y1 = 1 - cosine_similarity(q2, y1.detach(), dim=-1).mean()
        loss_q2sub1 = 1- cosine_similarity(q2, sub1.detach(), dim=-1).mean()

        loss = 1 / 2 * (args.alpha * loss_q1y2 + args.beta * loss_q1sub2
        + args.alpha * loss_q2y1 + args.beta * loss_q2sub1)

        loss.backward()

        # update online network
        optimizer.step()
        # update target network
        model.update_target_network(mm)

        return loss.detach().cpu().numpy()

    # @profile(precision=4)
    def eval(b):
        model.eval()
        if args.no_edge == False:
            b = trans2hyper(b, args.aug_ratio)
        b = add_zero(b)

        with torch.no_grad():
            dist_1, dist_2, dist_3, dist_4 = model.inference(b)
            dist = args.alpha * (dist_1 + dist_3)/2 + args.beta * (dist_2 + dist_4)/2
            ano_score = dist.cpu().numpy()

        scaler = MinMaxScaler()
        ano_score = scaler.fit_transform(ano_score.reshape(-1, 1)).reshape(-1)
        labels = b.ano_n_label[:b.batch_size]
        return ano_score, labels.cpu().numpy()


    # @profile(precision=4)
    def save_model(epoch):

        saved_content = {}

        saved_content['OurModel'] = model.state_dict()
        saved_content['epoch'] = epoch

        # torch.save(saved_content, 'checkpoint/{}/{}_{}.pth'.format(args.dataset,args.setting, gen_num))
        torch.save(saved_content, 'checkpoint/{}/model_NodePred_{}.pth'.format(args.dataset, args.epochs))

        return


    def load_model(filename):

        loaded_content = torch.load('checkpoint/{}/model_NodePred_{}.pth'.format(args.dataset, filename),
                                    map_location=lambda storage, loc: storage)

        model.load_state_dict(loaded_content['OurModel'])

        print("Loaded epoch number: " + str(loaded_content['epoch']))
        print("Successfully loaded!")

        return


    best_auc, best_precision, best_recall = 0., 0., 0.
    start_tr = time.time()
    for epoch in tqdm(range(1, args.epochs + 1)):
        l_ano_score, l_ano_label, l_loss = [], [], []
        for b in data_loader:
            # pre-processing: dual hypergraph transformation and perturbation
            b1, b2 = transform_1(b), transform_2(b)
            if args.no_edge == False:
                b1, b2 = trans2hyper(b1, args.aug_ratio), trans2hyper(b2, args.aug_ratio)
            b1, b2 = add_zero(b1), add_zero(b2)

            loss = train(epoch-1, b1, b2)
            l_loss.append(loss)
            if epoch % args.eval_epochs == 0 or epoch == args.epochs:
                ano_score, ano_label = eval(b)
                l_ano_score.append(ano_score)
                l_ano_label.append(ano_label)
        if epoch % args.eval_epochs == 0 or epoch == args.epochs:
            loss_avg = np.mean(l_loss)
            all_ano_score = np.concatenate(l_ano_score, axis=0)
            all_ano_label = np.concatenate(l_ano_label, axis=0)
            all_ano_score_b = np.array([1. if s > 0.5 else 0. for s in all_ano_score])

            if args.dataset == 'DGraphFin':
                recall, macro_f1, AUC, acc, precision = \
                    accuracy(all_ano_score, all_ano_label)
            else:
                recall, macro_f1, AUC, acc, precision = accuracy(all_ano_score, all_ano_label)
            if AUC > best_auc:
                best_auc = AUC
                save_model(epoch)
    end_tr = time.time()

    start_inf = time.time()
    print('\n========================Final Evaluation Results========================')
    auc_max, pre_max, recall_max = 0, 0, 0
    load_model(str(args.epochs))
    for i in tqdm(range(1, args.eval_rounds + 1)):
        l_ano_score, l_ano_label, l_loss = [], [], []
        for b in data_loader:
            ano_score, ano_label = eval(b)
            l_ano_score.append(ano_score)
            l_ano_label.append(ano_label)
        all_ano_score = np.concatenate(l_ano_score, axis=0)
        all_ano_label = np.concatenate(l_ano_label, axis=0)

        if args.dataset == 'DGraphFin':
            recall, macro_f1, AUC, acc, precision = \
                accuracy(all_ano_score, all_ano_label)
        else:
            recall, macro_f1, AUC, acc, precision = accuracy(all_ano_score, all_ano_label)

        if AUC > auc_max:
            saved_dict = {'ano_score': all_ano_score, 'ano_label': all_ano_label}
            np.save('./eval/{}/saved.npy'.format(args.dataset), saved_dict)
            auc_max = AUC
            pre_max = precision
            recall_max = recall
    end_inf = time.time()

    print('Best Evaluation AUC: ', auc_max)
    print('Precision: ', pre_max)
    print('Recall: ', recall_max)
    print("Training time for {} epochs: {}".format(str(args.epochs), str(end_tr-start_tr)))
    print("Inference time for {} epochs: {}".format(str(args.eval_rounds), str(end_inf - start_inf)))






