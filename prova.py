from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf


from file_utils import *
from utils import *
from neural_networks import GCN


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('type', 'gcn', 'learning method') # 'gcn', 'sage'
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
 


# Load data

###########OLD
features_old, labels_old, adj_old = load_cora()
xfeatures, xlabels, xadj = process_input_data(features_old, labels_old, adj_old)
#xy_train, xy_test, xy_val, xtrain_mask, xtest_mask, xval_mask = split_dataset (xlabels)
xfeatures = preprocess_features(xfeatures)
xsupport = [xpreprocess_adj(xadj)]


#yadj, yfeatures, yy_train, yy_val, yy_test, ytrain_mask, yval_mask, ytest_mask = load_data_old("cora")
###########OLD


####################   vedi questi se funzionano meglio ####################
features, adj, labels = load_data("cora")
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(labels)

#renormalization trick della matrice di adiacenza e conversione a tupla
support = [preprocess_adj(adj, True)]
features /= features.sum(1).reshape(-1, 1)
features = sparse_to_tuple(sp.csr_matrix(features))


num_supports = 1


GCN_placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for i in range(num_supports)],
    'feats': tf.sparse_placeholder(tf.float32, shape=tf.constant((2708, 1433), dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create network
network = GCN(GCN_placeholders, input_dim=1433)

# Initialize session
sess = tf.Session()
# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
train_dict = build_dictionary(features, support, y_train, train_mask, GCN_placeholders)
train_dict.update({GCN_placeholders['dropout']: FLAGS.dropout})
#validation_dict = build_dictionary(features, support, y_val, val_mask, GCN_placeholders)


def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = build_dictionary(features, support, labels, mask, GCN_placeholders)
    outs_val = sess.run([network.loss, network.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Train network
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    

    # Training step 
    train_out = sess.run([network.opt_op, network.loss, network.accuracy], feed_dict=train_dict)

    # Validation
    t_test = time.time()

    #cost, acc, duration = evaluate(features, support, y_val, val_mask, GCN_placeholders)
    #cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_out[1]),
          "train_acc=", "{:.5f}".format(train_out[2]), "val_loss=")
          #, "{:.5f}".format(cost),
          #"val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))



    
network.save(sess)

print("Optimization Finished!")


test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, GCN_placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))






















