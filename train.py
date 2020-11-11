import numpy as np
import scipy.sparse as spsparse
import torch
import torch.nn as nn
import os
from models.model import motif_emb
from utils import data_process
import networkx as nx
from scipy.sparse import csr_matrix
import torch.nn.functional as F
import argparse

PARSER = argparse.ArgumentParser(description='Parsing Input to Model')
PARSER.add_argument("--dropout", type=float, default=0.5)
PARSER.add_argument("--lr", type=float, default=0.01)
PARSER.add_argument("--dataset", type=str, default='cora')
PARSER.add_argument("--l2_coef", type=float, default=0.0)
PARSER.add_argument('--testing', type=float, default=0.6)
PARSER.add_argument('--epoch', type=int, default=50)
PARSER.add_argument('--batch', type=int, default=5)
PARSER.add_argument('--num_motif', type=int, default=20)
PARSER.add_argument('--pre_train', type=int, default=20)
PARSER.add_argument('--no_attention', dest='no_attention', default=True, action='store_false')
PARSER.add_argument('--motif', type=str, default="0123456")
PARSER.add_argument('--ud_motif', type=str, default='01')
PARSER.add_argument('--sample_epoch', type=int, default=1)
PARSER.add_argument('--no_skew', dest='no_skew', default=False, action='store_true')
PARSER.add_argument('--hidden', type=int, default=256)

ARGS = PARSER.parse_args()

ROOT = os.getcwd()
HIDDEN_SIZE = ARGS.hidden
DATASET = ARGS.dataset
DROPOUT = ARGS.dropout
TESTING = ARGS.testing
NO_SKEW = ARGS.no_skew
SAMPLE_EPOCH = ARGS.sample_epoch
ATTENTION_WEIGHT = ARGS.no_attention
cuda = False
if torch.cuda.is_available():
    cuda = True
n_batch = ARGS.batch
lr = ARGS.lr
l2_coef = ARGS.l2_coef
n_epochs = ARGS.epoch
if 'citeseer' in DATASET:
    n_epochs += 100
    lr = 0.001
if SAMPLE_EPOCH < 1:
    SAMPLE_EPOCH = n_epochs
PRE_TRAIN_EPOCH = ARGS.pre_train
random_seed = 12345
num_motif = ARGS.num_motif
SELECTED_MOTIFS = map(int, list(ARGS.motif))
MOTIF_LENGTH = data_process.get_cite_motif_length()
SELECTED_MOTIFS.sort()

torch.manual_seed(random_seed)
np.random.seed(random_seed)

data_process.set_directory(ROOT)

"""Weight on losses"""
alpha_list = [[0.7, 0.3]]

"""Preprocess of Graph, Label, Attribute"""
mask, labels, features, G = data_process.preprocess(dataset=ROOT + "/data/{}/".format(DATASET) + DATASET)
G = G.subgraph(max(nx.connected_component_subgraphs(G.copy().to_undirected()), key=len).nodes())
features = features[G.nodes()]
labels = labels[G.nodes()]
new_mask, _ = data_process.get_mask(G.nodes())
LCC_cites = data_process.LCC_cites(mask, new_mask, ROOT, DATASET)
features = data_process.preprocess_features(csr_matrix(features))

"""Motif mining/loading"""
MOTIF_PATH = ROOT + "/data/dicts/" + DATASET + "_multi_LCC"
if not os.path.isdir(MOTIF_PATH):
    M_type_dict, M_instance_dict = data_process.motif_mining(G, ROOT, SELECTED_MOTIFS, DATASET, mask=new_mask)
    print("Done Processing")
    data_process.save_dict(M_type_dict, M_instance_dict, DATASET, path=ROOT + "/data/dicts/")
    print("Saved")
else:
    print("Loading data")
    M_type_dict, M_instance_dict = data_process.load_dict(DATASET, path=ROOT + "/data/dicts/")
M_type_dict = torch.tensor(M_type_dict[:, SELECTED_MOTIFS])
M_instance_dict = {k: M_instance_dict[k] for k in SELECTED_MOTIFS}
print("MOTIFS: ", M_instance_dict.keys())
n_nodes = features.shape[0]
n_class = len(np.unique(labels))
n_motif = len(SELECTED_MOTIFS)

print("Number of Motif", n_motif)
n_feat = features.shape[1]
n_hid1 = HIDDEN_SIZE
n_hid2 = HIDDEN_SIZE
print("Number of Nodes: ", n_nodes)

labels = torch.LongTensor(labels)
features = torch.FloatTensor(features[np.newaxis])
G = G.to_undirected()
adj = nx.adjacency_matrix(G)
adj = data_process.normalize_adj(adj + spsparse.eye(adj.shape[0]))

sp_adj = data_process.sparse_mx_to_torch_sparse_tensor(adj)

"""Running"""
accuracys = []
for alpha in alpha_list:
    try:
        randoms = []
        if cuda:
            torch.cuda.empty_cache()
        model = motif_emb(n_feat, n_hid1, n_hid2, n_motif, n_class, DROPOUT)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

        train_index, val_index, test_index = data_process.train_test_split(n_nodes, TESTING)

        motif_indicator = torch.cat([M_type_dict, M_type_dict], 0).permute(1, 0).float()

        if cuda:
            torch.cuda.empty_cache()
            model = model.cuda()
            features = features.cuda()
            labels = labels.cuda()
            motif_indicator = motif_indicator.cuda()
            sp_adj = sp_adj.cuda()

        b_xent = nn.BCEWithLogitsLoss()
        supervised_loss = nn.CrossEntropyLoss(reduction='none')
        xent = nn.CrossEntropyLoss()

        best = -3

        batch_indices = None
        motif_nodes = None
        for epoch in range(n_epochs):
            """Get batch of motifs, and nodes correponding to motifs"""
            if epoch % SAMPLE_EPOCH is 0:
                batch_indices, motif_nodes = data_process.get_multiple_motif_batch(M_instance_dict,
                                                                                   range(n_nodes),
                                                                                   n_batch,
                                                                                   MOTIF_LENGTH,
                                                                                   num_motif)
            """Get Batch of training nodes"""
            node_indices = data_process.get_batch_nodes(len(train_index.tolist()), n_batch)
            dgi = True
            total_loss = []
            total_train_acc = []
            for i in range(n_batch):
                model.train()
                optimizer.zero_grad()
                # Negative Samples
                idx = np.random.permutation(n_nodes)
                shuf_fts = features[:, idx, :]
                """Nodes used in training """
                node_batch = torch.tensor(train_index[node_indices[i]])
                """Nodes used for unsupervised"""
                motif_node = torch.tensor(motif_nodes[i])
                batch_node = torch.cat([motif_node, motif_node], 0).squeeze()
                lbl_1 = torch.ones(1, len(motif_node))
                lbl_2 = torch.zeros(1, len(motif_node))
                lbl = torch.cat((lbl_1, lbl_2), 1)

                if cuda:
                    shuf_fts = shuf_fts.cuda()
                    lbl = lbl.cuda()
                logits, preds, weights = model(features, sp_adj, batch_indices[i], shuf_fts)

                """Novelty Weight(Skew-aware sample weighting)"""
                if NO_SKEW:
                    supervised_weight = torch.ones(node_batch.shape[0]).cuda()
                else:
                    supervised_weight = data_process.supervised_weight(weights, node_batch)

                """Task Weight(Node-sensitive Motif Regularization)"""
                if ATTENTION_WEIGHT:
                    curr_weight = torch.mul(weights, motif_indicator)[:, batch_node]
                else:
                    curr_weight = motif_indicator[:, batch_node]
                loss = 0
                for j in range(n_motif):
                    loss += F.binary_cross_entropy_with_logits(logits[j], lbl, curr_weight[j, :]) * alpha[0]

                if epoch > PRE_TRAIN_EPOCH:
                    loss2 = (supervised_loss(preds[node_batch], labels[node_batch]) * supervised_weight).mean()
                    loss = loss + loss2 * alpha[1]

                acc_train = torch.sum(torch.argmax(preds[node_batch], dim=1)
                                      == labels[node_batch]).float() / node_batch.shape[0]
                loss.backward(retain_graph=False)
                optimizer.step()
                total_loss.append(loss.detach())
                total_train_acc.append(acc_train.detach())

            total_loss = torch.tensor(total_loss)
            total_train_acc = torch.tensor(total_train_acc)
            predicts = torch.argmax(model.predict(features, sp_adj, val_index), dim=1)
            acc_val = torch.sum(predicts == labels[val_index]).float() / val_index.shape[0]
            if epoch % 5 == 0:
                if epoch <= PRE_TRAIN_EPOCH:
                    print("Epoch: {} | loss {:4.4f}"
                          .format(epoch, torch.mean(total_loss)))
                else:
                    print("Epoch: {} | loss: {:4.4f} | train acc: {:4.4f} | val acc: {:4.4f}"
                          .format(epoch, torch.mean(total_loss), torch.mean(total_train_acc), acc_val))
            if acc_val > best:
                best = acc_val
                torch.save(model.state_dict(),
                           (ROOT + '/results/' + DATASET + '_best_dgi_{}_{}.pkl').format(TESTING, DROPOUT))

        model.load_state_dict(
            torch.load((ROOT + '/results/' + DATASET + '_best_dgi_{}_{}.pkl').format(TESTING, DROPOUT)))
        predicts = torch.argmax(model.predict(features, sp_adj, test_index), dim=1)
        acc = torch.sum(predicts == labels[test_index]).float() / test_index.shape[0]
        randoms.append(str(acc.item()))
        print("TESTING ACCURACY: ", acc)
        accuracys.append(randoms)
    except KeyboardInterrupt:
        print("STOPPING EARLY")
        random = []
        model.load_state_dict(
            torch.load((ROOT + '/results/' + DATASET + '_best_dgi_{}_{}.pkl').format(TESTING, DROPOUT)))
        predicts = torch.argmax(model.predict(features, sp_adj, test_index), dim=1)
        acc = torch.sum(predicts == labels[test_index]).float() / test_index.shape[0]
        randoms.append(str(acc.item()))
        print("TESTING ACCURACY: ", acc)
        accuracys.append(randoms)
        break
print("ACCURACYS: ", accuracys)
means = data_process.get_mean(accuracys)
data_process.write_to_file(accuracys, DATASET, ROOT, TESTING, DROPOUT, means)
