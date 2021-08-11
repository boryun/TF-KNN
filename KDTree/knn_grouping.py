import os
import sys
import tensorflow as tf
from tensorflow.python.framework import ops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module = tf.load_op_library(os.path.join(BASE_DIR, "tf_knn_grouping.so"))


def knn_grouping(global_points, ref_points, k, omp=True):
    """
    Get the KNN indices for M reference points within N global points
    of B batches. (custom operation, cud based burte force)
    
    Args:
        global_points: [B, N, 3], whole point cloud.
        ref_points: [B, M, 3], query points.
        k: int, number of neighbours.
        omp: bool, switch for OpenMP parallel computation.
    Return:
        indices: [B, M, k], KNN index of M query points of each batch.
        dists: distenct for each KNN to the corresponding ref_point.
    """
    indices, dists = grouping_module.knn_grouping_ann(global_points, ref_points, k, omp=omp)
    dists = tf.sqrt(dists)
    return indices, dists  # using tf.gather with batch_dims=1 to gather NNs
ops.NoGradient('KnnGroupingANN')

