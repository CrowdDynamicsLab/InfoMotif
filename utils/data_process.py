import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso
import itertools
import os.path
import subprocess
import torch.nn.functional as F
import pickle
import os
import scipy.sparse as sp
import torch
import json
import scipy.io as sio

mapping = {"Case_Based": 0, "Genetic_Algorithms": 1, "Neural_Networks": 2, "Probabilistic_Methods": 3,
           "Reinforcement_Learning": 4, "Rule_Learning": 5, "Theory": 6}
mapping1 = {"Agents": 0, "IR": 1, "DB": 2, "AI": 3, "HCI": 4, "ML": 5}

motifs1 = {
    0: nx.DiGraph([(1, 2), (2, 3)]),
    1: nx.DiGraph([(1, 2), (1, 3)]),
    2: nx.DiGraph([(2, 1), (3, 1)]),
    3: nx.DiGraph([(1, 2), (2, 3), (1, 3)]),
    4: nx.DiGraph([(1, 2), (2, 3), (3, 2)]),
    5: nx.DiGraph([(1, 2), (1, 3), (3, 1)]),
    6: nx.DiGraph([(1, 2), (2, 3), (3, 1)]),
}


def set_directory(root):
    if not os.path.isdir(root + '/data/dicts'):
        os.mkdir(root + '/data/dicts')
    if not os.path.isdir(root + '/results'):
        os.mkdir(root + '/results')
    if not os.path.isdir(root + '/data/motif_instances'):
        os.mkdir(root + '/data/motif_instances')


def preprocess(dataset='data/cora', directed='directed', split=0.8):
    f = open(dataset + '.cites', 'r')
    mask, labels, attribute = get_dataset(dataset)
    G = None
    if directed == "directed":
        G = nx.DiGraph()
    for line in f.readlines():
        edge = line.strip().split()
        if edge[0] != edge[1]:
            G.add_edge(int(mask[edge[0]]), int(mask[edge[1]]))
    return mask, labels, attribute, G


def LCC_cites(mask, new_mask, ROOT, DATASET):
    f = open("{}/data/{}/{}.cites".format(ROOT, DATASET, DATASET))
    g = open("{}/data/{}/{}_LCC.cites".format(ROOT, DATASET, DATASET), 'w+')
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        if line[0] == line[1] or (mask[line[0]] not in new_mask or mask[line[1]] not in new_mask):
            continue
        g.write("{}\t{}\n".format(new_mask[mask[line[0]]], new_mask[mask[line[1]]]))
    g.close()


def get_dataset(dataset):
    f = open(dataset + ".content", 'r')
    lines = f.readlines()
    mask = {}
    labels = np.zeros(len(lines))
    if 'cora' in dataset:
        attribute = np.zeros((len(lines), 1433))
    elif 'citeseer' in dataset:
        attribute = np.zeros((len(lines), 3703))
    idx = 0
    for line in lines:
        line = line.strip().split()
        mask[line[0]] = idx
        attribute[idx] = np.array(map(float, line[1:-1]))
        if 'cora' in dataset:
            labels[idx] = mapping[line[-1]]
        elif 'citeseer' in dataset:
            labels[idx] = mapping1[line[-1]]

        idx += 1
    return mask, labels, attribute


def load_social_data(dataset):
    mat_contents = sio.loadmat(dataset + ".mat")
    adj = mat_contents["Network"].toarray()
    if not os.path.exists(dataset + '.cites'):
        f = open(dataset + '.cites', "w+")
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i][j] != 0:
                    f.write(str(i) + '\t' + str(j) + '\n')
        f.close()
    adj = np.array(adj, dtype=int)
    features = mat_contents["Attributes"]
    label = mat_contents["Label"]
    G = nx.convert_matrix.from_numpy_matrix(adj, parallel_edges=False, create_using=nx.Graph)
    return label.flatten() - 1, features, G


def get_social_motif_length(ROOT):
    motif_def = ROOT + '/data/motif_json/motif.json'
    f = open(motif_def, 'r+')
    motif_json = json.load(f)
    result = {}
    for key in motif_json.keys():
        result[int(key)] = len(motif_json[key]["v"])
    return result


def get_cite_motif_length():
    mo = motifs1
    result = {}
    for key in mo.keys():
        result[key] = len(list(mo[key].nodes()))
    return result


def motif_mining(gr, ROOT, mo, DATASET, mask):
    nodes = list(gr.nodes())
    M_type_dict = np.zeros((len(nodes), len(mo)))
    M_instance_dict = {i: [] for i in mo}
    for k, v in M_instance_dict.items():
        M_instance_dict[k] = {mask[node]: [] for node in nodes}
    motif_def = ROOT + '/data/motif_json/motif.json'
    result_path = '{}/data/motif_instances/{}_instances.txt'.format(ROOT, DATASET)
    curr_motif = 0
    for motif in mo:
        print("Mining motifs {}".format(str(motif)))
        calc_motif_submatch(str(motif), motif_def, ROOT, DATASET)
        f = open(result_path, 'r+')
        count = 0
        for line in f.readlines():
            line = line.strip().split(' ')
            motif_node = map(int, line)
            motif_node.sort()
            for curr_nodes in motif_node:
                if motif_node not in M_instance_dict[motif][curr_nodes]:
                    M_instance_dict[motif][curr_nodes].append(motif_node)
                    M_type_dict[curr_nodes][curr_motif] = 1
            count += 1
        curr_motif += 1
    return M_type_dict, M_instance_dict


def calc_motif_submatch(motif, motif_def, ROOT, dataset):
    '''Compute motif using subgraph matching'''
    graph_path = ROOT + '/data/{}/{}_LCC.cites'.format(dataset, dataset)
    motif_path = ROOT + '/data/motif_json/{}motif.json'.format(dataset)
    submatch_path = ROOT + "/utils/vflib/call_submatch.py"

    def submatch(motif_json):
        motif_json = {'1': motif_json}
        with open(motif_path, 'w') as f:
            json.dump(motif_json, f)
        try:
            print('Call subgraph match for motif {}'.format(str(motif)))
            subprocess.call(['python', submatch_path, '-G', graph_path, '-M', motif_path,
                             '-D', dataset, '-R', ROOT],
                            cwd=ROOT + '/utils/vflib/')
        except Exception as e:
            print(e)
            print('Subgraph match failed')
            exit(1)
        print('Parse results')

    with open(motif_def, 'r') as f:
        motif_json = json.load(f)
    try:
        motif_json = motif_json[motif]
    except KeyError as e:
        print('Motif ' + motif + ' definition not found!')
        exit(1)
    _ = motif_json.pop('m', None)
    submatch(motif_json)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def save_dict(M_type_dict, M_instance_dict, dataset, path="data/dicts/"):
    directory = path + dataset + '_multi_LCC/'
    if os.path.isdir(directory):
        f = open(directory + "M_type_dict.pkl", 'wb')
        pickle.dump(M_type_dict, f)
        f.close()
        f = open(directory + "M_instance_dict.pkl", 'wb')
        pickle.dump(M_instance_dict, f)
        f.close()
    else:
        os.mkdir(directory)
        f = open(directory + "M_type_dict.pkl", 'wb')
        pickle.dump(M_type_dict, f)
        f.close()
        f = open(directory + "M_instance_dict.pkl", 'wb')
        pickle.dump(M_instance_dict, f)
        f.close()


def load_dict(dataset, path="data/dicts/"):
    directory = path + dataset + "_multi_LCC/"
    M_type_dict = pickle.load(open(directory + "M_type_dict.pkl", 'rb'))
    M_instance_dict = pickle.load(open(directory + "M_instance_dict.pkl", 'rb'))

    return M_type_dict, M_instance_dict


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def process_large_data(m_instance, n_nodes, n_batch):
    output = []
    splits = np.array_split(np.random.permutation(range(len(m_instance))), n_batch)
    for i in splits:
        output.append(m_instance[i])
    return output


def get_mask(nodes):
    count = 0
    mask = {}
    reverse_mask = {}
    for node in nodes:
        mask[node] = count
        reverse_mask[count] = node
        count += 1
    return mask, reverse_mask


def get_batch_nodes(n_nodes, n_batch):
    splits = np.array_split(np.random.permutation(range(n_nodes)), n_batch)
    return splits


def get_motif_batch(M_instance_dict, nodes, MOTIF_LENGTH, num_motif):
    temp_batch = None
    for node in nodes:
        curr_motifs = M_instance_dict[node]
        if len(curr_motifs) > num_motif:
            index = np.random.permutation(range(len(curr_motifs)))[:num_motif]
        elif len(curr_motifs) > 0:
            index = np.random.randint(low=0, high=len(curr_motifs), size=num_motif)
        if len(curr_motifs) > 0:
            curr_batch = np.array(curr_motifs)[index][np.newaxis, :]  # 20 * 2, expanded to 1 * 20 * 2
        else:
            curr_batch = np.random.randint(low=0, high=5, size=(1, num_motif, MOTIF_LENGTH))
        if temp_batch is None:
            temp_batch = curr_batch
        else:
            temp_batch = np.vstack((temp_batch, curr_batch))
    return temp_batch


def get_multiple_motif_batch(M_instance_dict, nodes, n_batch, MOTIF_LENGTH, num_motif):
    batch_indices = [[]] * n_batch
    node_indices = get_batch_nodes(len(nodes), n_batch)
    numpy_nodes = np.array(nodes)
    for i in range(n_batch):
        result = []
        for j in M_instance_dict.keys():
            batch_node = get_motif_batch(M_instance_dict[j],
                                         numpy_nodes[node_indices[i]].tolist(), MOTIF_LENGTH[j],
                                         num_motif)
            result.append(batch_node)
        batch_indices[i] = result
    return batch_indices, node_indices


def get_index(motif_batch, batches):
    index = []
    for node in batches:
        temp_index = np.where(motif_batch[:, 0] == node)
        index.append(temp_index)
    return index


def train_test_split(n_nodes, testing):
    splits = np.array_split(np.random.permutation(range(n_nodes)), 20)
    n_testing = int(testing * 100 / 5)
    n_val = 4

    test_index = np.concatenate(splits[:n_testing], axis=0)
    val_index = torch.LongTensor(np.concatenate(splits[n_testing:n_testing + n_val], axis=0))
    train_index = torch.LongTensor(np.concatenate(splits[n_testing + n_val:], axis=0))
    return train_index, val_index, test_index


def write_to_file(accuracys, dataset, root, testing, dropout, means):
    f = open((root + '/results/' + dataset + "_result_" + str(testing) + "_{}.txt").format(dropout), "w+")
    for accuracy in accuracys:
        f.write("\t".join(accuracy) + '\n')
    for mean in means:
        f.write("MEAN: " + str(mean) + '\n')
    f.close()


def get_mean(accuracys):
    means = []
    for acc in accuracys:
        means.append(np.mean(np.array(acc, dtype=float)))
    return np.array(means).tolist()


def supervised_weight(weights, curr_train):
    # number of motifs * number of nodes
    curr_weight = weights[:, :weights.shape[1] / 2].squeeze()[:, curr_train]

    # number of nodes * 1
    avg_weight = torch.mean(curr_weight, dim=1).unsqueeze(1).expand_as(curr_weight)

    distance = torch.sum((curr_weight - avg_weight) ** 2, dim=0)
    distance_normalized = F.softmax(distance)
    return distance_normalized * curr_train.shape[0]
