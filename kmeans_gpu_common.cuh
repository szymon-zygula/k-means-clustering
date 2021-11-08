#ifndef KMEANS_GPU_COMMON_H
#define KMEANS_GPU_COMMON_H

#include <cstdlib>

#include "kmeans_device_data.cuh"
#include "config.cuh"

#define DOUBLE_INFINITY (1.0 / 0.0)

namespace kmeans_gpu {
    template<size_t dim>
    __device__
    void copy_centroids_to_shared(
        common::DeviceDataRaw<dim>& data,
        double* shared_centroids
    ) {
        size_t iterations =
            (data.centroid_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        for(size_t i = 0; i < dim; ++i) {
            size_t dim_offset = i * data.centroid_count;
            for(size_t j = 0; j < iterations; ++j) {
                size_t centr_idx = THREADS_PER_BLOCK * j + threadIdx.x;
                if(centr_idx >= data.centroid_count) {
                    break;
                }

                size_t idx = centr_idx + dim_offset;
                shared_centroids[idx] = data.d_centroids[idx];
            }
        }

        __syncwarp();
        __syncthreads();
    }

    template<size_t dim>
    __device__
    inline void update_deltas(
        common::DeviceDataRaw<dim>& data,
        size_t vec_idx,
        size_t closest_centroid
    ) {
        if(data.d_memberships[vec_idx] != closest_centroid) {
            data.d_memberships[vec_idx] = closest_centroid;
            data.d_deltas[vec_idx] = 1.0;
        }
    }

    template<size_t dim>
    __device__
    double calc_centroid_distance(
        common::DeviceDataRaw<dim>& data,
        size_t vec_idx,
        size_t centroid,
        double* shared_centroids,
        double* object_coords
    ) {
        double centroid_coord = DeviceVecArray<dim>::get(
            shared_centroids, data.centroid_count, centroid, dim - 1
        );

        double coord_diff = object_coords[dim - 1] - centroid_coord;

        double tail = calc_centroid_distance<dim - 1>(
            *(common::DeviceDataRaw<dim - 1>*)&data,
            vec_idx,
            centroid,
            shared_centroids,
            object_coords
        );

        return coord_diff * coord_diff + tail;
    }

    template<>
    __device__
    inline double calc_centroid_distance<0>(
        common::DeviceDataRaw<0>& data,
        size_t vec_idx,
        size_t centroid,
        double* shared_centroids,
        double* object_coords
    ) {
        return 0.0;
    }

    template<size_t dim>
    __device__
    size_t find_closest_centroid(
        common::DeviceDataRaw<dim>& data,
        size_t vec_idx,
        double* shared_centroids
    ) {
        size_t closest_centroid;
        double min_dist = DOUBLE_INFINITY;

        double object_coords[dim];
        for(size_t i = 0; i < dim; ++i) {
            object_coords[i] = DeviceVecArray<dim>::get(
                data.d_objects, data.object_count, data.d_object_permutation[vec_idx], i
            );
        }

        for(size_t i = 0; i < data.centroid_count; ++i) {
            double dist =
                calc_centroid_distance<dim>(data, vec_idx, i, shared_centroids, object_coords);
            if(dist < min_dist) {
                min_dist = dist;
                closest_centroid = i;
            }
        }

        return closest_centroid;
    }


    double reduce_deltas(thrust::device_vector<double>& d_deltas);
}

#endif
