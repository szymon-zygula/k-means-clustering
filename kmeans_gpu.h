#ifndef KMEANS_CLUSTERING_GPU_H
#define KMEANS_CLUSTERING_GPU_H

#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include "vec.h"
#include "kmeans_device_data.h"
#include "device_vec_array.h"

#define DOUBLE_INFINITY (1.0 / 0.0)

namespace kmeans_gpu {
    static constexpr size_t THREADS_PER_BLOCK = 1024;

    template<size_t dim>
    __device__
    double calc_centroid_distance(DeviceDataRaw data, size_t vec_idx, size_t centroid) {
        double dist = 0.0;
        for(size_t j = 0; j < dim; ++j) {
            double coord_diff =
                DeviceVecArray<dim>::get(data.d_objects, data.object_count, vec_idx, j) -
                DeviceVecArray<dim>::get(data.d_centroids, data.centroid_count, centroid, j);
            dist += coord_diff * coord_diff;
        }

        return dist;
    }

    template<size_t dim>
    __device__
    size_t find_closest_centroid(DeviceDataRaw data, size_t vec_idx) {
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

    __device__
    void update_deltas(DeviceDataRaw data, size_t vec_idx, size_t closest_centroid) {
        if(data.d_memberships[vec_idx] != closest_centroid) {
            data.d_memberships[vec_idx] = closest_centroid;
            data.d_deltas[vec_idx] = 1.0;
        }
    }

    template<size_t dim>
    __device__
    void update_new_centroids(DeviceDataRaw data, size_t vec_idx, size_t closest_centroid) {
        for(size_t i = 0; i < dim; ++i) {
            atomicAdd(
                &DeviceVecArray<dim>::get(data.d_new_centroids, data.centroid_count, closest_centroid, i),
                DeviceVecArray<dim>::get(data.d_objects, data.object_count, vec_idx, i)
            );
        }

        atomicAdd(&data.d_new_cluster_sizes[closest_centroid], 1u);
    }

    template<size_t dim>
    __global__
    void assign_to_closest_centroid(DeviceDataRaw data) {
        size_t vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(vec_idx >= data.object_count) {
            return;
        }

        size_t closest_centroid = find_closest_centroid<dim>(data, vec_idx);

        update_deltas(data, vec_idx, closest_centroid);
        update_new_centroids<dim>(data, vec_idx, closest_centroid);
    }

    double reduce_deltas(thrust::device_vector<double>& d_deltas) {
        double delta = thrust::reduce(d_deltas.begin(), d_deltas.end(), 0.0);
        thrust::fill(d_deltas.begin(), d_deltas.end(), 0.0);
        return delta;
    }

    template<size_t dim>
    double calculate_nearest_centroids(DeviceData<dim>& data) {
        size_t block_count = (data.d_objects.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        assign_to_closest_centroid<dim><<<block_count, THREADS_PER_BLOCK>>>(data.to_raw_pointers());
        return reduce_deltas(data.d_deltas);
    }

    template<size_t dim>
    __global__
    void update_centroids_ker(DeviceDataRaw data) {
        size_t centroid_idx = threadIdx.x + blockIdx.x * blockDim.x;

        if(centroid_idx >= data.centroid_count) {
            return;
        }

        double divisor = data.d_new_cluster_sizes[centroid_idx] == 0
            ? 1.0
            : data.d_new_cluster_sizes[centroid_idx];
        for(size_t i = 0; i < dim; ++i) {
            DeviceVecArray<dim>::get(data.d_centroids, data.centroid_count, centroid_idx, i) =
                DeviceVecArray<dim>::get(data.d_new_centroids, data.centroid_count, centroid_idx, i) / divisor;
            DeviceVecArray<dim>::get(data.d_new_centroids, data.centroid_count, centroid_idx, i) = 0.0;
        }

        data.d_new_cluster_sizes[centroid_idx] = 0;
    }

    template<size_t dim>
    void update_centroids(DeviceData<dim>& data) {
        size_t block_count = (data.d_centroids.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        update_centroids_ker<dim><<<block_count, THREADS_PER_BLOCK>>>(data.to_raw_pointers());
    }

    template<size_t dim>
    thrust::host_vector<size_t> kmeans_clustering(
        thrust::host_vector<kmeans::Vec<dim>>& h_centroids,
        thrust::host_vector<kmeans::Vec<dim>>& h_objects
    ) {
        double delta = std::numeric_limits<double>::infinity();
        DeviceData<dim> data(h_centroids, h_objects);

        while(delta / h_objects.size() > kmeans::ACCURACY_THRESHOLD) {
            delta = calculate_nearest_centroids(data);
            cudaDeviceSynchronize();
            update_centroids(data);
        }

        h_centroids = data.get_host_centroids();
        return data.get_host_memberships();
    }
}

#endif
