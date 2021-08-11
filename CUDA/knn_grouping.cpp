#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cuda_runtime.h>

using namespace tensorflow;

REGISTER_OP("KnnGroupingCUDA")
    .Input("global_points: float32")
    .Input("query_points: float32")
    .Attr("K: int")
    .Output("idx: int32")
    .Output("dist: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        ::tensorflow::shape_inference::ShapeHandle dims1, dims2;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &dims2));
        int K;
        TF_RETURN_IF_ERROR(c->GetAttr("K", &K));
        ::tensorflow::shape_inference::ShapeHandle output_shape = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), K});
        c->set_output(0, output_shape);
        c->set_output(1, output_shape);
        return Status::OK();
    });

void knn_kernel_launcher(int B, int N, int M, int K, int dim, const float* globals, const float* queries, int* index, float* dist, float* dist_cache);
class KnnGroupingCUDAOp: public OpKernel {
  public:
    KnnGroupingCUDAOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("K", &K_));
        OP_REQUIRES(context, K_ > 0, errors::InvalidArgument("Expect positive K!"));
    }

    void Compute(OpKernelContext* context) override {
        // acquire & check inputs
        const Tensor& global_points = context->input(0);
        OP_REQUIRES(
            context, 
            global_points.dims()==3, 
            errors::InvalidArgument("global_points should have shape of (B,N,3)!")
        );
        OP_REQUIRES(
            context, 
            K_ <= global_points.dim_size(1), 
            errors::InvalidArgument("K is bigger than the total number of points in global_points!")
        );

        const Tensor& query_points = context->input(1);
        OP_REQUIRES(
            context, 
            query_points.dims()==3, 
            errors::InvalidArgument("query_points should have shape of (B,N,D)!")
        );
        OP_REQUIRES(
            context,
            query_points.dim_size(2) == global_points.dim_size(2),
            errors::InvalidArgument("query_points should have same dims as global_points.")
        );

        // shape params
        const int B = global_points.dim_size(0);
        const int N = global_points.dim_size(1);
        const int M = query_points.dim_size(1);
        const int K = K_;
        const int D = global_points.dim_size(2);
        
        // create output tensor
        Tensor* nn_idx = NULL;
        Tensor* nn_dist = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,M,K_}, &nn_idx));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{B,M,K_}, &nn_dist));

        // dist_cache for per-batch KNN calculation
        Tensor dist_cache;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, TensorShape{M,N}, &dist_cache));

        auto g_pts = global_points.flat<float>();
        auto q_pts = query_points.flat<float>();
        auto idx = nn_idx->flat<int>();
        auto dist = nn_dist->flat<float>();
        auto cache = dist_cache.flat<float>();
        knn_kernel_launcher(B, N, M, K_, D, &g_pts(0), &q_pts(0), &idx(0), &dist(0), &cache(0));
    }
  private:
    int K_;
};

REGISTER_KERNEL_BUILDER(Name("KnnGroupingCUDA").Device(DEVICE_GPU), KnnGroupingCUDAOp);
