from __future__ import division
from __future__ import print_function

from tmp_net import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'ENZYMES', 'which dataset to load') #ENZYMES, PROTEINS
flags.DEFINE_boolean('featureless', False, 'If nodes are featureless')


if FLAGS.dataset=='ENZYMES':
    num_nodes = 19580
    num_graphs = 600
    tot = 20180
    num_classes = 6
    num_feats = 18
    dataset_name = "enzymes"
    splits = [[0,580], [580, 590], [590, 600]]
elif FLAGS.dataset=='FRANKENSTEIN': #non entra in memoria con 16gb di ram
    num_nodes = 73283
    num_graphs = 4337
    num_classes = 2
    num_feats = 780
    splits = [[0,2000], [2000, 2500], [2500, 4337]]
    dataset_name = "frankenstein"
elif FLAGS.dataset=='PROTEINS': #questo sì
    num_nodes = 43471
    num_graphs = 1113
    tot = 44584
    num_classes = 2
    num_feats = 29
    splits = [[0,1000], [1000, 1050], [1050, 1100]]
    dataset_name = "proteins"

num_test = 100


acc=[0. for i in range(0, num_test)]

for i in range(0,num_test):
    print("test num: {} running ...".format(i))
    acc[i] = test(num_nodes, num_graphs, num_classes, num_feats, dataset_name, splits, FLAGS.featureless)

mean_acc = np.max(acc)
std_dev_acc = np.std(acc)
print("mean accuracy on {} dataset:  {}".format(FLAGS.dataset, mean_acc))
print("std dev: {}".format(std_dev_acc))
