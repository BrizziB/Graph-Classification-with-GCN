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


def load_data(path="cora", dataset="cora"):
    print('Loading: {} dataset...'.format(dataset))

    feats_and_labels = np.genfromtxt("{}\{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(feats_and_labels[:, 1:-1], dtype=np.int32)
    labels = encode_onehot(feats_and_labels[:, -1])

    # costruzione grafo
    idx = np.array(feats_and_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}\{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.int32)

    # simmetrizzazione della matrice d'adiacenza
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    print('{} has {} nodes, {} edges, {} features.'.format(dataset, adj.shape[0], edges.shape[0], features.shape[1]))

    return features, adj, labels

def get_splits(labels):
    train_ind = range(140)
    val_ind = range(200, 500)
    test_ind = range(500, 1500)

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
