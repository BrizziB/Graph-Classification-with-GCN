import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import sys

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return onehot


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset, model):
    if model == 'gcn':
        return load_data_gcn(dataset)



def load_data_gcn(dataset="cora"):
    print('Loading: {} dataset...'.format(dataset))

    feats_and_labels = np.genfromtxt("cora\{}.content".format(dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(feats_and_labels[:, 1:-1], dtype=np.int32)
    labels = encode_onehot(feats_and_labels[:, -1])

    # costruzione grafo
    idx = np.array(feats_and_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("cora\{}.cites".format(dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.int32)

    # simmetrizzazione della matrice d'adiacenza
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    print('{} has {} nodes, {} edges, {} features.'.format(dataset, adj.shape[0], edges.shape[0], features.shape[1]))

    return features, adj, labels

def get_splits(labels, train_dim, val_dim, test_dim):
    train_ind = range(train_dim[1])
    val_ind = range(val_dim[0], val_dim[1])
    test_ind = range(test_dim[0], test_dim[1])

    labels_train = np.zeros(labels.shape, dtype=np.int32)
    labels_val = np.zeros(labels.shape, dtype=np.int32)
    labels_test = np.zeros(labels.shape, dtype=np.int32)

    labels_train[train_ind] = labels[train_ind]
    labels_val[val_ind] = labels[val_ind]
    labels_test[test_ind] = labels[test_ind]

    train_mask = sample_mask(train_ind, labels.shape[0])
    val_mask = sample_mask(val_ind, labels.shape[0])
    test_mask = sample_mask(test_ind, labels.shape[0])

    return labels_train, labels_val, labels_test, train_ind, val_ind, test_ind, train_mask, val_mask, test_mask


def load_data_ENZYMES():
    num_nodes = 19580
    num_graphs = 600
    sparse_adj = build_adj_diag(num_nodes, path="enzymes/ENZYMES_A.txt")
    node_feats = build_feats_vertConc(path="enzymes/ENZYMES_node_attributes.txt")
    graph_labels = build_labels_vertConc(num_graphs, path="enzymes/ENZYMES_graph_labels.txt") #codifica one-hot
    return sparse_adj, node_feats, graph_labels


def build_labels_vertConc(graphs, path):
    labels = encode_onehot(np.loadtxt(path, dtype='i', delimiter=','))
    print(labels)
    return labels

def build_feats_vertConc(path):
    feats_matrix = np.loadtxt(path, delimiter=',')
    #print(feats_matrix)
    return sp.csr_matrix(feats_matrix)


#dal file ENZYMES_A costruisce una matrice diagonale a blocchi contenente le matrici di adiacenza di ogni grafo (un grafo per blocco)
def build_adj_diag(nodes, path):
    tmpdata = np.genfromtxt(path, dtype=np.dtype(str))
    ind1 = tmpdata[:, 1]
    ind2 = tmpdata[:, 0]
    adj_matrix = [[0 for i in range(nodes)] for k in range(nodes)]
    for i in range(len(ind1)):
        u = ind1[i]
        v = ind2[i]
        u = int(u)       #vanno letti come stringhe
        v = int(v[:-1]) #aggiustamenti per eliminare la virgola del file 
        adj_matrix[u-1][v-1] = 1 

    adj_matrix = np.matrix(adj_matrix)
    #sparse_adj = sp.csr_matrix(adj_matrix)
    #print(sparse_adj)
    #np.savetxt("tmpMat.txt", adjMatrix, fmt='%4.0f')
    return sp.coo_matrix(adj_matrix)
