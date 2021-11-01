#ifndef KMEANS_GPU_COMMON_H
#define KMEANS_GPU_COMMON_H

#include <cstdlib>

#include "kmeans_device_data.cuh"

#define DOUBLE_INFINITY (1.0 / 0.0)

namespace kmeans_gpu {
    static constexpr size_t THREADS_PER_BLOCK = 1024;

    template<size_t dim>
    __device__
    inline void update_deltas(
        common::DeviceDataRaw<dim> data,
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
        common::DeviceDataRaw<dim> data,
        size_t vec_idx,
        size_t centroid
    ) {
        double dist = 0.0;
        for(size_t j = 0; j < dim; ++j) {
            double object_coord = DeviceVecArray<dim>::get(
                data.d_objects, data.object_count, data.d_object_permutation[vec_idx], j
            );
            double centroid_coord = DeviceVecArray<dim>::get(
                data.d_centroids, data.centroid_count, centroid, j
            );
            double coord_diff = object_coord - centroid_coord;
            dist += coord_diff * coord_diff;
        }

        return dist;
    }

    template<size_t dim>
    __device__
    size_t find_closest_centroid(
        common::DeviceDataRaw<dim> data,
        size_t vec_idx
    ) {
        size_t closest_centroid;
        double min_dist = DOUBLE_INFINITY;
        for(size_t i = 0; i < data.centroid_count; ++i) {
            double dist = calc_centroid_distance<dim>(data, vec_idx, i);
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
