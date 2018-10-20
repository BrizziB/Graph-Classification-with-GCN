import numpy as np
import time
import scipy.sparse as sp
import tensorflow as tf



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
def process_features(features):
    features /= features.sum(1).reshape(-1, 1)
    return sparse_to_tuple(sp.csr_matrix(features))

#renormalization trick della matrice di adiacenza
def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return sp.csr_matrix(a_norm)


#conversione a tupla e normalizzazione della matrice d'adiacenza
def preprocess_adj(adj, is_gcn, symmetric = True):
    if is_gcn:
        adj = adj + sp.eye(adj.shape[0]) # ogni nodo ha come vicino anche se stesso, fa parte di GCN
    adj = normalize_adj(adj, symmetric)
    return sparse_to_tuple(adj)


#  --------------------- metriche --------------------------------------------
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



#  ----------------------- init -----------------------------------------------
#inizializzatore di pesi secondo Glorot&Bengio  - vedi come funziona 
def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    val = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(val, name=name)

def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

#costruzione del dizionario per GCN 
def build_dictionary_GCN(feats, support, labels, labels_mask, placeholders):
    #prepara il dizionario che sarà poi passato alla sessione di TF
    dictionary = dict()
    dictionary.update({placeholders['labels']: labels})
    dictionary.update({placeholders['labels_mask']: labels_mask})
    dictionary.update({placeholders['feats']: feats})
    dictionary.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    dictionary.update({placeholders['num_features_nonzero']: feats[1].shape})
    return dictionary

#costruzione del dizionario per Sage 
def build_dictionary_Sage(feats, support, labels, labels_mask, degree, placeholders):
    #prepara il dizionario che sarà poi passato alla sessione di TF
    dictionary = dict()
    dictionary.update({placeholders['labels']: labels})
    dictionary.update({placeholders['labels_mask']: labels_mask})
    dictionary.update({placeholders['feats']: feats})
    dictionary.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    dictionary.update({placeholders['num_features_nonzero']: feats[1].shape})
    dictionary.update({placeholders['degree']: degree})
    return dictionary
