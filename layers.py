import tensorflow as tf
import keras as k
from utils import *

_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    def __init__(self, **kwargs):
        layer = self.__class__.__name__.lower()
        name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.weights = {}
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

    def log_weights(self):
        for w in self.weights:
            tf.summary.histogram(self.name + '/weights/' + w, self.weights[w])

class ConvolutionalLayer(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout,
                 sparse_inputs, activation, isLast=False, bias=False, featureless=False, **kwargs):
        super(ConvolutionalLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.featureless = featureless
        self.activation = activation
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_weights'):
            for i in range(len(self.support)):
                self.weights['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.weights['bias'] = zeros([output_dim], name='bias')

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless: 
                pre_sup = dot(x, self.weights['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.weights['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.weights['bias']

        return self.activation(output)


class MaxLayer(Layer):
    def __init__(self, **kwargs):
        super(MaxLayer, self).__init__(**kwargs)

    def _call(self, inputs):
        x = inputs
        return tf.reduce_max(x, axis=0, keepdims=False)



class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, dropout, sparse_inputs,
                 placeholders=None, activation=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.dropout=0.5

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.activation = activation
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        #self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_weights'):
            self.weights['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.weights['bias'] = zeros([output_dim], name='bias')


    def _call(self, inputs):
        x = inputs

        # applico il dropout
        #if self.sparse_inputs:
        #    x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        #else:
        x = tf.nn.dropout(x, 1-self.dropout)

        # la moltiplicazione fra features e pesi - in questo consiste il layer dense
        output = dot(x, self.weights['weights'], sparse=self.sparse_inputs)

        # eventualmente applico il bias sommandolo all'output
        if self.bias:
            output += self.weights['bias']

        return self.activation(output) #l'uscita passa prima per la funzione di attivazione - una relu

class MeanPoolingLayer(Layer):
    """ Aggregates via mean-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, placeholders=None,
            dropout=False, bias=False, activation=tf.nn.relu, name=None, **kwargs):
        super(MeanPoolingLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.activation = activation
        self.support = placeholders['support']
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_weights'):
            for i in range(len(self.support)):
                self.weights['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.weights['bias'] = zeros([output_dim], name='bias')

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        
        
        self_vecs = inputs
        neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size, self.neigh_input_dim))

        for layer in self.mlp_layers: #filtro con il dense layer e poi faccio pooling
            h_reshaped = layer(h_reshaped)

        neigh_h = tf.reshape(h_reshaped, (batch_size, 1, self.hidden_dim))
        neigh_h = tf.reduce_mean(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.weights['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.weights["self_weights"])
        
        #if not self.concat:
        output = tf.add_n([from_self, from_neighs])
        #else:
        #output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.weights['bias']
       
        return self.activation(output)
    
    
    
    
    
    
    
    
    
    
    
    
    
    