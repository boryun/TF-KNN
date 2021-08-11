import time
import numpy as np
import tensorflow as tf
from CUDA.knn_grouping import knn_grouping as cuda_grouping
from KDTree.knn_grouping import knn_grouping as kdtree_grouping

def tf_knn_grouping(global_points, ref_points, k, sort=False):
    # calculate the distance matrix
    A2 = tf.reduce_sum(ref_points * ref_points, axis=2, keepdims=True)  # [B, M, 1], x1^2 + y1^2 + z1^2
    B2 = tf.reduce_sum(global_points * global_points, axis=2, keepdims=True)  # [B, N, 1], x2^2 + y2^2 + z2^2
    AB = tf.matmul(ref_points, tf.transpose(global_points, perm=[0, 2, 1]))  # [B, M, N], x1*x2 + y1*y2 + z1*z2
    dist_matrix = A2 - 2*AB + tf.transpose(B2, perm=[0, 2, 1])  # [B, M, N]

    # get top-k indices
    dist, indices = tf.nn.top_k(-dist_matrix, k=k, sorted=sort)  # [B, M, k(indices within N)]
    dist = tf.sqrt(dist * -1)  # the dist_matrix was timed by -1 so top_k will return K nearest
    
    return indices, dist  # using tf.gather with batch_dims=1


if __name__ == "__main__":
    import numpy as np
    import time

    def get_mx(*shape):
        return tf.convert_to_tensor(np.random.uniform(-30, 30, shape).astype(np.float32))
    
    B = 8
    N = 8192
    M = 512
    K = 16

    g_pts = get_mx(B, N, 3)
    r_pts = get_mx(B, M, 3)

    # warm up
    _, _ = tf_knn_grouping(get_mx(2, 512, 3), get_mx(2, 128, 3), 16)

    # pure tf version
    try:
        total = []
        for _ in range(20):
            t0 = time.time()
            idx0, dist0 = tf_knn_grouping(g_pts, r_pts, K)
            idx0, dist0 = idx0.numpy(), dist0.numpy()
            t1 = time.time()
            total.append(t1 - t0)
        print("pure_knn:", np.mean(total))
    except tf.errors.ResourceExhaustedError:  # OOM
        print("pure_knn: OOM")

    # cuda version
    total = []
    for _ in range(20):
        t0 = time.time()
        idx1, dist1 = cuda_grouping(g_pts, r_pts, K)
        idx1, dist1 = idx1.numpy(), dist1.numpy()
        t1 = time.time()
        total.append(t1 - t0)
    print("cuda_knn:", np.mean(total))

    # KDTree version, dominator
    total = []
    for _ in range(20):
        t0 = time.time()
        idx2, dist2 = kdtree_grouping(g_pts, r_pts, K, omp=False)
        idx2, dist2 = idx2.numpy(), dist2.numpy()
        t1 = time.time()
        total.append(t1 - t0)
    print("kdtre_knn(omp off):", np.mean(total))

    # KDTree version, dominator
    total = []
    for _ in range(20):
        t0 = time.time()
        idx2, dist2 = kdtree_grouping(g_pts, r_pts, K, omp=True)
        idx2, dist2 = idx2.numpy(), dist2.numpy()
        t1 = time.time()
        total.append(t1 - t0)
    print("kdtre_knn(omp on):", np.mean(total))
