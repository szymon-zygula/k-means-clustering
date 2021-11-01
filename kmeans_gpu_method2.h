#ifndef KMEANS_GPU_METHOD2
#define KMEANS_GPU_METHOD2

#include <thrust/sort.h>
#include <thrust/reduce.h>

#include "kmeans_gpu_common.h"

namespace kmeans_gpu {
    namespace method2 {
        template<size_t dim>
        __global__
        void assign_to_closest_centroid(DeviceDataRaw<dim> data) {
            size_t vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
            if(vec_idx >= data.object_count) {
                return;
            }

            size_t closest_centroid = find_closest_centroid<dim>(data, vec_idx);

            update_deltas<dim>(data, vec_idx, closest_centroid);
        }

        template<size_t dim>
        double calculate_nearest_centroids(DeviceData<dim>& data) {
            size_t block_count = (data.d_objects.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            assign_to_closest_centroid<dim><<<
                block_count, THREADS_PER_BLOCK
            >>>(
                data.to_raw_pointers()
            );

            cudaDeviceSynchronize();

            return reduce_deltas(data.d_deltas);
        }

        template<size_t dim>
        __global__
        void calculate_new_centroids(DeviceDataRaw<dim> data) {
            size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

            if(tid >= data.reduced_count) {
                return;
            }

            size_t centroid_idx = data.d_reduced_memberships[tid];

            for(size_t i = 0; i < dim; ++i) {
                double* centroid_coord = &DeviceVecArray<dim>::get(
                    data.d_centroids, data.centroid_count, centroid_idx, i
                );

                *centroid_coord = data.d_reduced_objects[tid].coords[i] / data.d_reduced_counts[tid];
            }
        }

        template<size_t dim>
        void update_centroids(DeviceData<dim>& data) {
            thrust::sort_by_key(
                data.d_memberships.begin(), data.d_memberships.end(), data.d_aos_objects.begin()
            );

            // TODO: Get rid of this
            auto temp = static_cast<thrust::host_vector<kmeans::Vec<dim>>>(data.d_aos_objects);
            data.d_objects = temp;

            auto new_end = thrust::reduce_by_key(
                data.d_memberships.begin(), data.d_memberships.end(), data.d_aos_objects.begin(),
                data.d_reduced_memberships.begin(), data.d_reduced_objects.begin()
            );

            thrust::reduce_by_key(
                data.d_memberships.begin(), data.d_memberships.end(), data.d_ones.begin(),
                data.d_reduced_memberships.begin(), data.d_reduced_counts.begin()
            );

            size_t reduced_count = new_end.first - data.d_reduced_memberships.begin();
            DeviceDataRaw<dim> raw_data = data.to_raw_pointers();
            raw_data.reduced_count = reduced_count;

            size_t block_count = (reduced_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            calculate_new_centroids<dim><<<block_count, THREADS_PER_BLOCK>>>(raw_data);
        }

        #include "kmeans_clustering_gpu.h"
    }
}

#endif
