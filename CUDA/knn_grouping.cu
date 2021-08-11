#include <cuda.h>
#include <cassert>

#define BLOCK_DIM 16  // along dim-axis


__global__ void compute_squared_distance(const float * __restrict__ global_pts, int global_num, const float * __restrict__ query_pts, int query_num, int dim, float * __restrict__ dist_cache) {    
    // sliding block for global matrix and query matrix
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // offset of sliding block to the begining of overall matrix
    __shared__ int base_A;  // A -> global
    __shared__ int base_B;  // B -> query
    base_A = blockIdx.x * BLOCK_DIM;
    base_B = blockIdx.y * BLOCK_DIM;

    // index of current thread
    int dx = threadIdx.x;
    int dy = threadIdx.y;
    
    //! dx and dy have different meaning in loading and calculating step, so we need 2 pairs of conditions
    int load_valid_A = (base_A + dy < global_num);
    int load_valid_B = (base_B + dy < query_num);
    int cal_valid_A = load_valid_A;
    int cal_valid_B = (base_B + dx < query_num);

    float ssd = 0.0f;  // sliding squared distance

    // slide the block along the dim axis to accumulate the squared distance
    for (int base_dim = 0; base_dim < dim; base_dim += BLOCK_DIM) {

        // STEP-1: Load data to shared matrix from global and query that mapped to the sliding block
        // In this case, [dy, dx] indicate a unique cell of the sliding block, so that each 
        // thread load one value to shared matrix.
        int dim_offset = base_dim + dx;
        if(dim_offset < dim) {  // if current cell not exceed dim axis
            shared_A[dy][dx] = load_valid_A ? global_pts[(base_A + dy) * dim + dim_offset] : 0.0f;
            shared_B[dy][dx] = load_valid_B ? query_pts[(base_B + dy) * dim + dim_offset] : 0.0f;
        }
        else {  // otherwise
            shared_A[dy][dx] = 0.0f;
            shared_B[dy][dx] = 0.0f;
        }
        __syncthreads();

        // STEP-2: Accumulating the squared distance at current block position.
        // In this case, a thread with index of (dy, dx) accumulate the distance between dy-th 
        // global point and dx-th query point of current block.
        if (cal_valid_A && cal_valid_B) {
            for(int i = 0; i < BLOCK_DIM; i++) {
                float dif = shared_A[dy][i] - shared_B[dx][i];
                ssd += dif * dif;
            }
        }
        __syncthreads();
    }

    // STEP-2': write the accumulated squared distance to dist matrix
    if(cal_valid_A && cal_valid_B) {
        int idx_A = base_A + dy;
        int idx_B = base_B + dx;
        dist_cache[idx_B *global_num + idx_A] = ssd;
    }
}

__global__ void insertion_sort(int global_num, int query_num, int k, int * __restrict__ index, float * __restrict__ dist_cache){
    // index of query point handled by current thread
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pointIdx < query_num) {
        // move pointers to the begining of current query point
        int * p_index = index + pointIdx * k;
        float * p_dist = dist_cache + pointIdx * global_num;

        // init the first index
        p_index[0] = 0;

        // traverse all global points
        for (int i = 0; i < global_num; i++) {
            int cur_index = i;
            float cur_dist = p_dist[i];

            // skip if current distance is bigger than **seened** k-th smallest dist
            if(i >= k && cur_dist >= p_dist[k-1]) {
                continue;
            }

            // insert cur_dist to k-th smallest ary
            int j = min(i, k - 1);
            while(j > 0 && p_dist[j-1] > cur_dist) {
                // shift right to move space for cur_value
                p_index[j] = p_index[j-1];
                p_dist[j] = p_dist[j-1];
                j--;
            }
            // insert cur_value
            p_index[j] = cur_index;
            p_dist[j] = cur_dist;
        }
    }
}

__global__ void compute_and_copy_root_square(int global_num, int query_num, int k, float * __restrict__ dist_cache) {
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int neighborIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if(pointIdx < query_num && neighborIdx < k) {
        int pos = pointIdx * k + neighborIdx;
        dist_cache[pos] = sqrtf(dist_cache[pos]);
    }
}

void knn_kernel_launcher(int B, int N, int M, int K, int dim, const float * __restrict__ globals, const float * __restrict__ queries, int * __restrict__ index, float * __restrict__ dist, float * __restrict__ dist_cache) {
    for(int batch_idx = 0; batch_idx < B; batch_idx++) {
        // Move pointers to current batch
        const float * batch_globals = globals + batch_idx * N * dim;
        const float * batch_queries = queries + batch_idx * M * dim;
        int * batch_index = index + batch_idx * M * K;
        float * batch_dist = dist + batch_idx * M * K;

        // Compute the squared euclidean distances
        dim3 grid0(N / BLOCK_DIM + (N % BLOCK_DIM != 0), M / BLOCK_DIM + (M % BLOCK_DIM != 0), 1);
        dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
        compute_squared_distance<<<grid0, block0>>>(batch_globals, N, batch_queries, M, dim, dist_cache);

        // Sort the distances with their respective indexes
        dim3 grid1(M / 256 + (M % 256 != 0), 1, 1);
        dim3 block1(256, 1, 1);
        insertion_sort<<<grid1, block1>>>(N, M, K, batch_index, dist_cache);

        // Compute the square root of the k smallest distances
        dim3 grid2(M / 16 + (M % 16 != 0), K / 16 + (K % 16 != 0), 1);
        dim3 block2(16, 16, 1);
        compute_and_copy_root_square<<<grid2, block2>>>(N, M, K, dist_cache);

        // copy dist from cache to output
        int FLOAT_SIZE = sizeof(float);
        cudaMemcpy2D(batch_dist, M*K*FLOAT_SIZE, dist_cache, M*N*FLOAT_SIZE, K, M, cudaMemcpyDeviceToDevice);
    }
}
