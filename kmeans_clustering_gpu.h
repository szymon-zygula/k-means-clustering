#ifndef KMEANS_CLUSTERING_GPU_H
#define KMEANS_CLUSTERING_GPU_H

#include <iostream>
#include <limits>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include "kmeans_clustering_common.h"

namespace kmeans_gpu {
    static constexpr size_t THREADS_PER_BLOCK = 1024;

    template<size_t dim>
    class VecArray {
        size_t _size;
        double* d_array;

        public:
        VecArray(size_t size) {
            _size = size;
            cudaMalloc(&d_array, sizeof(double) * _size * dim);
            cudaMemset(d_array, 0, sizeof(double) * _size * dim);
        }

        VecArray(thrust::host_vector<kmeans::Vec<dim>> objects) {
            _size = objects.size();
            thrust::host_vector<double> rearranged_objects(_size * dim);

            for(size_t i = 0; i < dim; ++i) {
                for(size_t j = 0; j < _size; ++j) {
                    rearranged_objects[j + i * _size] = objects[j].coords[i];
                }
            }

            cudaMalloc(&d_array, sizeof(double) * _size * dim);
            cudaMemcpy(
                d_array, rearranged_objects.data(),
                sizeof(double) * _size * dim,
                cudaMemcpyHostToDevice
            );
        }

        size_t size() {
            return _size;
        }

        double* raw_data() {
            return d_array;
        }

        __device__
        static double& get(double* d_array, size_t size, size_t vec_idx, size_t dim_idx) {
            return d_array[vec_idx + dim_idx * size];
        }

        thrust::host_vector<kmeans::Vec<dim>> to_host() {
            thrust::host_vector<double> objects(_size * dim);
            cudaMemcpy(
                objects.data(), d_array,
                sizeof(double) * _size * dim,
                cudaMemcpyDeviceToHost
            );

            thrust::host_vector<kmeans::Vec<dim>> rearranged_objects(_size);

            for(size_t i = 0; i < dim; ++i) {
                for(size_t j = 0; j < _size; ++j) {
                    rearranged_objects[j].coords[i] = objects[j + i * _size];
                }
            }

            return rearranged_objects;
        }

        ~VecArray() {
            cudaFree(d_array);
        }
    };

    template<size_t dim>
    std::pair<kmeans::Vec<dim>, kmeans::Vec<dim>> calculate_bounding_box(
        thrust::host_vector<kmeans::Vec<dim>>& objects
    ) {
        kmeans::Vec<dim> low_bounding_box;
        kmeans::Vec<dim> high_bounding_box;

        for(size_t i = 0; i < dim; ++i) {
            double min = std::numeric_limits<double>::infinity();
            double max = -std::numeric_limits<double>::infinity();
            for(size_t j = 0; j < objects.size(); ++j) {
                min = objects[j].coords[i] < min ? objects[j].coords[i] : min;
                max = objects[j].coords[i] > max ? objects[j].coords[i] : max;
            }

            low_bounding_box.coords[i] = min;
            high_bounding_box.coords[i] = max;
        }

        return std::make_pair(low_bounding_box, high_bounding_box);
    }

    // TODO: Unify with CPU code, allow to be chosen before kmeans_clustering
    template<size_t dim>
    thrust::host_vector<kmeans::Vec<dim>> initialize_centroids(
        size_t k,
        thrust::host_vector<kmeans::Vec<dim>>& objects
    ) {
        auto bounding_box = calculate_bounding_box(objects);
        thrust::host_vector<kmeans::Vec<dim>> centroids(k);
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937_64 mersenne_twister(seed);

        for(size_t i = 0; i < dim; ++i) {
            double low = bounding_box.first.coords[i];
            double high = bounding_box.second.coords[i];
            std::uniform_real_distribution<double> centroid_distribution(low, high);
            for(size_t j = 0; j < centroids.size(); ++j) {
                centroids[j].coords[i] = centroid_distribution(mersenne_twister);
            }
        }

        return centroids;
    }

    template<size_t dim>
    __global__ void assign_to_closest_centroid(
        size_t object_count,
        double* d_objects,
        size_t centroid_count,
        double* d_centroids,
        double* d_new_centroids,
        unsigned* d_new_cluster_sizes,
        size_t* d_memberships,
        double* d_deltas
    ) {
        size_t vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(vec_idx >= object_count) {
            return;
        }

        size_t closest_centroid;
        double min_dist = 1.0 / 0.0; // Infinity
        for(size_t i = 0; i < centroid_count; ++i) {
            double dist = 0.0;
            for(size_t j = 0; j < dim; ++j) {
                double coord_diff =
                    VecArray<dim>::get(d_objects, object_count, vec_idx, j) -
                    VecArray<dim>::get(d_centroids, centroid_count, i, j);
                dist += coord_diff * coord_diff;
            }

            if(dist < min_dist) {
                min_dist = dist;
                closest_centroid = i;
            }
        }

        if(d_memberships[vec_idx] != closest_centroid) {
            d_memberships[vec_idx] = closest_centroid;
            d_deltas[vec_idx] = 1.0;
        }

        for(size_t i = 0; i < dim; ++i) {
            atomicAdd(
                &VecArray<dim>::get(d_new_centroids, centroid_count, closest_centroid, i),
                VecArray<dim>::get(d_objects, object_count, vec_idx, i)
            );
        }

        atomicAdd(&d_new_cluster_sizes[closest_centroid], 1u);
    }

    template<size_t dim>
    double calculate_nearest_centroids(
        VecArray<dim>& d_objects,
        VecArray<dim>& d_centroids,
        VecArray<dim>& d_new_centroids,
        thrust::device_vector<unsigned>& d_new_cluster_sizes,
        thrust::device_vector<size_t>& d_memberships,
        thrust::device_vector<double>& d_deltas
    ) {
        size_t block_count = (d_objects.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        assign_to_closest_centroid<dim><<<block_count, THREADS_PER_BLOCK>>>(
            d_objects.size(),
            d_objects.raw_data(),
            d_centroids.size(),
            d_centroids.raw_data(),
            d_new_centroids.raw_data(),
            thrust::raw_pointer_cast(d_new_cluster_sizes.data()),
            thrust::raw_pointer_cast(d_memberships.data()),
            thrust::raw_pointer_cast(d_deltas.data())
        );

        double delta = thrust::reduce(d_deltas.begin(), d_deltas.end(), 0.0);
        thrust::fill(d_deltas.begin(), d_deltas.end(), 0.0);
        return delta;
    }

    template<size_t dim>
    __global__ void update_centroids_ker(
        size_t centroid_count,
        double* d_centroids,
        double* d_new_centroids,
        unsigned* d_new_cluster_sizes
    ) {
        size_t centroid_idx = threadIdx.x + blockIdx.x * blockDim.x;

        if(centroid_idx >= centroid_count) {
            return;
        }

        double divisor = d_new_cluster_sizes[centroid_idx] == 0
            ? 1.0
            : d_new_cluster_sizes[centroid_idx];
        for(size_t i = 0; i < dim; ++i) {
            VecArray<dim>::get(d_centroids, centroid_count, centroid_idx, i) =
                VecArray<dim>::get(d_new_centroids, centroid_count, centroid_idx, i) / divisor;
            VecArray<dim>::get(d_new_centroids, centroid_count, centroid_idx, i) = 0.0;
        }

        d_new_cluster_sizes[centroid_idx] = 0;
    }

    template<size_t dim>
    void update_centroids(
        VecArray<dim>& d_centroids,
        VecArray<dim>& d_new_centroids,
        thrust::device_vector<unsigned>& d_new_cluster_sizes
    ) {
        size_t block_count = (d_centroids.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        update_centroids_ker<dim><<<block_count, THREADS_PER_BLOCK>>>(
            d_centroids.size(),
            d_centroids.raw_data(),
            d_new_centroids.raw_data(),
            thrust::raw_pointer_cast(d_new_cluster_sizes.data())
        );
    }

    template<size_t dim>
    std::pair<thrust::host_vector<kmeans::Vec<dim>>, thrust::host_vector<size_t>> kmeans_clustering(
        thrust::host_vector<kmeans::Vec<dim>>& h_objects, int k
    ) {
        double delta = std::numeric_limits<double>::infinity();

        thrust::host_vector<kmeans::Vec<dim>> h_centroids = initialize_centroids(k, h_objects);
        VecArray<dim> d_centroids(h_centroids);
        VecArray<dim> d_objects(h_objects);
        VecArray<dim> d_new_centroids(h_centroids.size());
        thrust::device_vector<unsigned> d_new_cluster_sizes(h_objects.size());
        thrust::device_vector<size_t> d_memberships(h_objects.size());
        thrust::device_vector<double> d_deltas(h_objects.size());

        int it = 0;
        while(delta / h_objects.size() > kmeans::ACCURACY_THRESHOLD) {
            std::cout << "iteration " << ++it << " (delta = " << delta << ")" << std::endl;
            delta = calculate_nearest_centroids(
                d_objects,
                d_centroids,
                d_new_centroids,
                d_new_cluster_sizes,
                d_memberships,
                d_deltas
            );

            cudaDeviceSynchronize();

            update_centroids(d_centroids, d_new_centroids, d_new_cluster_sizes);
        }

        return std::make_pair(d_centroids.to_host(), static_cast<thrust::host_vector<size_t>>(d_memberships));
    }
}

#endif
