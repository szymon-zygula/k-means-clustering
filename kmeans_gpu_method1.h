#ifndef KMEANS_GPU_METHOD1
#define KMEANS_GPU_METHOD1

#include "kmeans_gpu_common.h"

namespace kmeans_gpu {
    namespace method1 {
        template<size_t dim>
        __device__
        void update_new_centroids(DeviceDataRaw<dim> data, size_t vec_idx, size_t closest_centroid) {
            for(size_t i = 0; i < dim; ++i) {
                double* centr_coord = &DeviceVecArray<dim>::get(
                    data.d_new_centroids, data.centroid_count, closest_centroid, i
                );
                double obj_coord = DeviceVecArray<dim>::get(
                    data.d_objects, data.object_count, vec_idx, i
                );
                atomicAdd(centr_coord, obj_coord);
            }

            atomicAdd(&data.d_new_cluster_sizes[closest_centroid], 1u);
        }

        template<size_t dim>
        __global__
        void assign_to_closest_centroid(DeviceDataRaw<dim> data) {
            size_t vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
            if(vec_idx >= data.object_count) {
                return;
            }

            size_t closest_centroid = find_closest_centroid<dim>(data, vec_idx);

            update_deltas<dim>(data, vec_idx, closest_centroid);
            update_new_centroids<dim>(data, vec_idx, closest_centroid);
        }

        template<size_t dim>
        double calculate_nearest_centroids(DeviceData<dim>& data) {
            size_t block_count =
                (data.d_objects.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            assign_to_closest_centroid<dim><<<
                block_count, THREADS_PER_BLOCK
            >>>(
                data.to_raw_pointers()
            );

            return reduce_deltas(data.d_deltas);
        }

        template<size_t dim>
        __global__
        void update_centroids_ker(DeviceDataRaw<dim> data) {
            size_t centroid_idx = threadIdx.x + blockIdx.x * blockDim.x;

            if(centroid_idx >= data.centroid_count) {
                return;
            }

            double divisor = data.d_new_cluster_sizes[centroid_idx] == 0
                ? 1.0
                : data.d_new_cluster_sizes[centroid_idx];
            for(size_t i = 0; i < dim; ++i) {
                double* centr_coord = &DeviceVecArray<dim>::get(
                    data.d_centroids, data.centroid_count, centroid_idx, i
                );
                double* new_centr_coord = &DeviceVecArray<dim>::get(
                    data.d_new_centroids, data.centroid_count, centroid_idx, i
                );

                *centr_coord = *new_centr_coord / divisor;
                *new_centr_coord = 0.0;
            }

            data.d_new_cluster_sizes[centroid_idx] = 0;
        }

        template<size_t dim>
        void update_centroids(DeviceData<dim>& data) {
            size_t block_count =
                (data.d_centroids.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            update_centroids_ker<dim><<<
                block_count, THREADS_PER_BLOCK
            >>>(
                data.to_raw_pointers()
            );
        }

        #include "kmeans_clustering_gpu.h"
    }
}

#endif
