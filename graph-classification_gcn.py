from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import matplotlib.pyplot as plt

from file_utils import *
from utils import *
from neural_networks import GCNGraphs
from neural_networks import GCN
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'ENZYMES', 'which dataset to load') #ENZYMES, PROTEINS
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_boolean('featureless', False, 'If nodes are featureless')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')


if FLAGS.dataset=='ENZYMES':
    num_nodes = 19580
    num_graphs = 600
    tot = 20180
    num_classes = 6
    num_feats = 18
    dataset_name = "enzymes"
    splits = [[0,560], [590, 590], [560, 600]]

elif FLAGS.dataset=='PROTEINS': #questo sì
    num_nodes = 43471
    num_graphs = 1113
    tot = 44584
    num_classes = 2
    num_feats = 29
    splits = [[0,700], [900, 900], [700, 1113]]
    dataset_name = "proteins"

#print ("training on: {} dataset").format(FLAGS.dataset)

adj, features, labels, idx = load_data(num_nodes, num_graphs, num_classes, num_feats, dataset_name) #poi passalo al normalizzatore
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, val_mask, test_mask = get_splits_graphs(num_graphs, labels, splits[0], splits[1], splits[2], idx)


#adj, features, labels, idx = load_data_basic(num_nodes, num_graphs, num_classes, num_feats, dataset_name) 
#y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, val_mask, test_mask = get_splits_graphs_basic(num_graphs, labels, splits[0], splits[1], splits[2], idx)

support = [preprocess_adj(adj, True, False)]
features = process_features(features)

num_supports = 1

GCN_placeholders = {
    'idx' :tf.placeholder(tf.int32),
    'support': [tf.sparse_placeholder(tf.float32) for i in range(num_supports)],
    'feats': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)), 
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
}

if FLAGS.featureless:
    input_dim = tot
else:
    input_dim = features[2][1]

# Create network
network = GCNGraphs(GCN_placeholders, input_dim, featureless=(FLAGS.featureless) )

# Initialize session
sess = tf.Session()
# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
train_dict = build_dictionary_GCN(features, support, y_train, train_mask, GCN_placeholders)
train_dict.update({GCN_placeholders['dropout']: FLAGS.dropout})

train_loss = [0. for i in range(0, FLAGS.epochs)]
val_loss = [0. for i in range(0, FLAGS.epochs)]

def evaluate(features, support, labels, mask, placeholders): #pare che anche qui si cambino i pesi, quale loss usa ?
    t_test = time.time()
    feed_dict_val = build_dictionary_GCN(features, support, labels, mask, GCN_placeholders)
    outs_val = sess.run([network.loss, network.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Train network
for epoch in range(FLAGS.epochs):
    t = time.time()

    # Training step 
    train_out = sess.run([network.opt_op, network.loss, network.accuracy, network.outputs], feed_dict=train_dict)

    train_loss[epoch] = train_out[1]

    # Validation
    t_test = time.time()

    cost, acc, duration = evaluate(features, support, y_val, val_mask, GCN_placeholders)
    cost_val.append(cost)

    val_loss[epoch] = cost

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_out[1]),
          "train_acc=", "{:.5f}".format(train_out[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    """ if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break """ 

network.save(sess)

print("Optimization Finished!")


test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, GCN_placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

epochs = [i for i in range(0, FLAGS.epochs)]
plt.plot(np.array(epochs), np.array(train_loss), color='g')
plt.plot(np.array(epochs), np.array(val_loss), color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train loss over epochs with {} dataset'.format(dataset_name))
plt.show()


