import numpy as np
import time
import scipy.sparse as sp
import tensorflow as tf
from scipy.sparse.linalg.eigen.arpack import eigsh

#trasforma matrici in tuple
def to_tuple(mat):
    if not sp.isspmatrix_coo(mat):
        mat = mat.tocoo()
    idxs = np.vstack((mat.row, mat.col)).transpose()
    values = mat.data
    shape = mat.shape
    return idxs, values, shape


#trasforma matrici sparse in tuble
def sparse_to_tuple(sparse_mat):
    if isinstance(sparse_mat, list):
        for i in range(len(sparse_mat)):
            sparse_mat[i] = to_tuple(sparse_mat[i])
    else:
        sparse_mat = to_tuple(sparse_mat)

    return sparse_mat

#normalizza la matrice delle feature per riga e la trasforma in tupla
def preprocess_features(features):
    rowadd = np.array(features.sum(1))
    inv = np.power(rowadd, -1).flatten()
    inv[np.isinf(inv)] = 0. # casi di elem infiniti
    mat_inv = sp.diags(inv)
    features = mat_inv.dot(features)
    return sparse_to_tuple(features)


    
def xpreprocess_adj(adj):
    adj = adj + sp.eye(adj.shape[0]) # A' = A + I
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D - matrice degree
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-1/2
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # casi di elem infiniti
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-1/2 A' D^-1/2
    return sparse_to_tuple(adj_normalized)

# =============================================================================================================
# =============================================================================================================
# =============================================================================================================

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return sp.csr_matrix(a_norm)



#renormalization trick della matrice di adiacenza e conversione a tupla
def preprocess_adj(adj, symmetric = True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return sparse_to_tuple(adj)

# =============================================================================================================
# =============================================================================================================
# =============================================================================================================

#metriche

#cross-entropy con mascheramento per isolare i nodi con label
def masked_cross_entropy(predictions, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask) #per normalizzare la loss finale
    loss *= mask
    return tf.reduce_mean(loss)

#accuracy con mascheramento
def masked_accuracy(predictions, labels, mask):
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


#inizializzatore di pesi secondo Glorot&Bengio
def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    val = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(val, name=name)

def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

#costruzione dei dizionari per l'addestramento 
def build_dictionary(feats, support, labels, labels_mask, placeholders):
    #prepara il dizionario che sarà poi passato alla sessione di TF
    dictionary = dict()
    dictionary.update({placeholders['labels']: labels})
    dictionary.update({placeholders['labels_mask']: labels_mask})
    dictionary.update({placeholders['feats']: feats})
    dictionary.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    dictionary.update({placeholders['num_features_nonzero']: feats[1].shape})
    return dictionary
