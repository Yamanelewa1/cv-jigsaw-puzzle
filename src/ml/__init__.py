"""Milestone 2 -- learned compatibility for jigsaw reconstruction.

Two machine-learning models replace the hand-designed compatibility measure
of Milestone 1, while everything around them stays identical: the same
segmentation, the same piece description, and above all the **same assembly
algorithm**, so the comparison is between matchers rather than pipelines.

=========================  ===============================================
:mod:`src.ml.features`     a described side -> tensors (strip / vector)
:mod:`src.ml.dataset`      labelled positive and negative side pairs,
                           augmentation, and the split shared by both models
:mod:`src.ml.siamese`      model 1: Siamese convolutional network
:mod:`src.ml.gnn`          model 2: graph neural network over the puzzle
:mod:`src.ml.train`        training loops, identical protocol for both
:mod:`src.ml.infer`        model scores -> a Milestone 1 CompatibilityTable
:mod:`src.ml.evaluate`     the three-way comparison the brief asks for
=========================  ===============================================

The two models are different in kind, not in size.  The Siamese network
convolves over a pair of side strips in isolation; the graph network message-
passes over the whole puzzle at once, on a flat descriptor and with no
convolution anywhere, so a side is scored in the context of its rivals.
"""

from . import dataset, evaluate, features, gnn, infer, siamese, train

__all__ = ["features", "dataset", "siamese", "gnn", "train", "infer",
           "evaluate"]
