# TF_KNN

This repository contains TensorFlow KNN Ops based on CPU(KDTree) and GPU(CUDA) respectively. CUDA version is a modification of [KNN-CUDA](https://github.com/vincentfpgarcia/kNN-CUDA), KDTree version is based on [nanoflann](https://github.com/jlblancoc/nanoflann). 

There is also a pure tensorflow implementation of KNN in `demo.py`, yet it can easily get OOM when handling large scale pointcloud (basically the reason why I create this repository) as a distance matrix with the size of \[batch_size, num_points, num_queries\] need to be stored in VRAM (however, it is still faster than mine CUDA implementation when available, 🥺).

Notes:
- The GPU version is very sensitive to K, larger K may cause the computation time grows dramatically. 
- The CPU version is the dominator during time test, which I insist to use instead of CUDA or pure TensorFlow version.

# Usage

Both version has the same way to build Op (yet for GPU version you may need to change the `CUDA_HOME` in `compile.sh` first):
1. run `compile.sh` in CUDA or KDTree folder.
2. import the knn_grouping function in `knn_grouping.py`.
3. use `tf.gather` with `batch_dims=1' to gather the NN via returned indices.

# Time Consumption (in python)

Sys & Env Info

- SYS: Ubuntu 20.04.2 LTS
- CPU: Intel i7-6700
- GPU: Nvidia GTX980
- CUDA: 10.1
- Python: 3.7.9
- Tensorflow: 2.3.1

Denote B,N,M,K for batch_size, reference_points, query_points and num_neighbours respectively, the computation times in python (averaged over 20 runs) are as following:

**At**&nbsp;&nbsp;B=8, N=8192, M=512, K=16

| | Pure TF | CUDA | KDTree | KDTree(OpenMP) |
|:-:|:-:|:-:|:-:|:-:|
|time(s)|0.026|0.085|0.018|0.005|

**At**&nbsp;&nbsp;B=8, N=8192, M=512, K=32

| | Pure TF | CUDA | KDTree | KDTree(OpenMP) |
|:-:|:-:|:-:|:-:|:-:|
|time(ms)|0.035|0.202|0.025|0.006|

**At**&nbsp;&nbsp;B=8, N=8192, M=512, K=64

| | Pure TF | CUDA | KDTree | KDTree(OpenMP) |
|:-:|:-:|:-:|:-:|:-:|
|time(ms)|0.053|0.559|0.037|0.009|

**At**&nbsp;&nbsp;B=8, N=32768, M=2048, K=16

| | Pure TF | CUDA | KDTree | KDTree(OpenMP) |
|:-:|:-:|:-:|:-:|:-:|
|time(ms)|OOM|0.371|0.084|0.017|

**At**&nbsp;&nbsp;B=8, N=32768, M=2048, K=32

| | Pure TF | CUDA | KDTree | KDTree(OpenMP) |
|:-:|:-:|:-:|:-:|:-:|
|time(ms)|OOM|0.506|0.112|0.024|

**At**&nbsp;&nbsp;B=8, N=32768, M=2048, K=64

| | Pure TF | CUDA | KDTree | KDTree(OpenMP) |
|:-:|:-:|:-:|:-:|:-:|
|time(ms)|OOM|1.085|0.168|0.035|