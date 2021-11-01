#include "kmeans_gpu_common.cuh"

namespace kmeans_gpu {
    double reduce_deltas(thrust::device_vector<double>& d_deltas) {
        double delta = thrust::reduce(d_deltas.begin(), d_deltas.end(), 0.0);
        thrust::fill(d_deltas.begin(), d_deltas.end(), 0.0);
        return delta;
    }
}
