Based on 
  the article: 
      Semi-Supervised Classification with Graph Convolutional Networks - Thomas N. Kipf, Max Welling.
      link: https://arxiv.org/abs/1609.02907

  code: 
      https://github.com/tkipf/gcn.


This Tensorflow implementation of GCN tries to handle graph-classification tasks in two (similar) ways:

1. "global node" approach - for each graph, a global node is added, only global nodes are classified
2. global mean pooling approacah - added a mean-pooling layer as the last layer (before softmax) in GCN

Both approaches start with a sparse, block-diagonal version of an adjacency matrix that gathers all the graphs in the dataset.
See https://github.com/tkipf/gcn/issues/4#issuecomment-274445114  for a better explanation - with figures

Results: on Protein dataset -with the settings in the code- i obtain an average classification accuracy of 78.8% with 2.7% as standard deviation (the two approaches yield very close results).
This accuracy is an average of 100 result.
Each test has been conducted using 1000 graphs in the training set and 100 in the test set - with random splits.
