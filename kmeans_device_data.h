#ifndef KMEANS_DEVICE_DATA_H
#define KMEANS_DEVICE_DATA_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "vec.h"
#include "device_vec_array.h"

namespace kmeans_gpu {
    struct DeviceDataRaw {
        size_t object_count;
        double* d_objects;
        size_t centroid_count;
        double* d_centroids;
        double* d_new_centroids;
        unsigned* d_new_cluster_sizes;
        size_t* d_memberships;
        double* d_deltas;
    };

    template<size_t dim>
    struct DeviceData {
        DeviceVecArray<dim> d_objects;
        DeviceVecArray<dim> d_centroids;
        DeviceVecArray<dim> d_new_centroids;
        thrust::device_vector<unsigned> d_new_cluster_sizes;
        thrust::device_vector<size_t> d_memberships;
        thrust::device_vector<double> d_deltas;

        DeviceData(
            thrust::host_vector<kmeans::Vec<dim>>& h_centroids,
            thrust::host_vector<kmeans::Vec<dim>>& h_objects
        ) :
            d_centroids(h_centroids),
            d_objects(h_objects),
            d_new_centroids(h_centroids.size()),
            d_new_cluster_sizes(h_objects.size()),
            d_memberships(h_objects.size()),
            d_deltas(h_objects.size())
        {}

        DeviceDataRaw to_raw_pointers() {
            return {
                .object_count = d_objects.size(),
                .d_objects = d_objects.raw_data(),
                .centroid_count = d_centroids.size(),
                .d_centroids = d_centroids.raw_data(),
                .d_new_centroids = d_new_centroids.raw_data(),
                .d_new_cluster_sizes = thrust::raw_pointer_cast(d_new_cluster_sizes.data()),
                .d_memberships = thrust::raw_pointer_cast(d_memberships.data()),
                .d_deltas = thrust::raw_pointer_cast(d_deltas.data())
            };
        }

        thrust::host_vector<size_t> get_host_memberships() {
            return static_cast<thrust::host_vector<size_t>>(d_memberships);
        }

        thrust::host_vector<kmeans::Vec<dim>> get_host_centroids() {
            return d_centroids.to_host();
        }
    };

}

#endif
