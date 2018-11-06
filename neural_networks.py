from layers import *
from utils import *
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


#versione base del modello di neural network, ripreso dalla definizione usata in Keras
#sarebbe una classe astratta, molti metodi vanno implementati e alcune variabili inizializzate
class BaseNet(object):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__.lower()
        self.weights = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()
# costruzione del modello con layers generici
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

# salvo per comodità le variabili del modello invece che tenerle solo in tf.GraphKeys.GlOBALVARIABLES
        self.weights = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)}

# inizializzo loss e accuracy
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError


#se non li usi levali
    def save(self, sess):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.weights)
        print(self.weights)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.weights)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


#estende la rete generica
class GCN(BaseNet):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['feats']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].weights.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        #cross entropy loss
        self.loss += masked_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self): 

        self.layers.append(ConvolutionalLayer(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            activation=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            featureless=False))

        self.layers.append(ConvolutionalLayer(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            activation=lambda x: x,
                                            dropout=True,
                                            sparse_inputs=False))
             

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCNGraphs(BaseNet):
    def __init__(self, placeholders, input_dim, featureless, **kwargs):
        super(GCNGraphs, self).__init__(**kwargs)

        self.inputs = placeholders['feats']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.featureless = featureless
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].weights.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        #cross entropy loss dopo aver applicato un softmax layer
        self.loss += masked_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self): 
        self.layers.append(ConvolutionalLayer(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            activation=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            featureless = self.featureless))


        self.layers.append(ConvolutionalLayer(input_dim=FLAGS.hidden2,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            activation=lambda x: x,
                                            dropout=True,
                                            sparse_inputs=False,
                                            featureless = False))

        




    def predict(self):
        return tf.nn.softmax(self.outputs)