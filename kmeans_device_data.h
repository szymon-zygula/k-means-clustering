#ifndef KMEANS_DEVICE_DATA_H
#define KMEANS_DEVICE_DATA_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "vec.h"
#include "device_vec_array.h"

namespace kmeans_gpu {
    namespace common {
        template<size_t dim>
        struct DeviceDataRaw {
            size_t object_count;
            double* d_objects;
            size_t centroid_count;
            double* d_centroids;
            size_t* d_memberships;
            double* d_deltas;
        };
    }

    namespace method1 {
        template<size_t dim>
        struct DeviceDataRaw : public common::DeviceDataRaw<dim> {
            double* d_new_centroids;
            unsigned* d_new_cluster_sizes;
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

            DeviceDataRaw<dim> to_raw_pointers() {
                DeviceDataRaw<dim> raw_data;
                raw_data.object_count = d_objects.size();
                raw_data.d_objects = d_objects.raw_data();
                raw_data.centroid_count = d_centroids.size();
                raw_data.d_centroids = d_centroids.raw_data();
                raw_data.d_new_centroids = d_new_centroids.raw_data();
                raw_data.d_new_cluster_sizes = thrust::raw_pointer_cast(d_new_cluster_sizes.data());
                raw_data.d_memberships = thrust::raw_pointer_cast(d_memberships.data());
                raw_data.d_deltas = thrust::raw_pointer_cast(d_deltas.data());
                return raw_data;
            }

            thrust::host_vector<size_t> get_host_memberships() {
                return static_cast<thrust::host_vector<size_t>>(d_memberships);
            }

            thrust::host_vector<kmeans::Vec<dim>> get_host_centroids() {
                return d_centroids.to_host();
            }
        };
    }

    namespace method2 {
        template<size_t dim>
        struct DeviceDataRaw : public common::DeviceDataRaw<dim> {
            size_t reduced_count;
            kmeans::Vec<dim>* d_reduced_objects;
            size_t* d_reduced_memberships;
            size_t* d_reduced_counts;
        };

        template<size_t dim>
        struct DeviceData {
            thrust::host_vector<kmeans::Vec<dim>>* h_objects;
            thrust::host_vector<kmeans::Vec<dim>>* h_centroids;
            DeviceVecArray<dim> d_objects;
            DeviceVecArray<dim> d_centroids;
            thrust::device_vector<size_t> d_memberships;
            thrust::device_vector<double> d_deltas;

            thrust::device_vector<kmeans::Vec<dim>> d_aos_objects;
            thrust::device_vector<kmeans::Vec<dim>> d_reduced_objects;
            thrust::device_vector<size_t> d_reduced_memberships;
            thrust::device_vector<size_t> d_reduced_counts;

            thrust::device_vector<size_t> d_ones;

            DeviceData(
                thrust::host_vector<kmeans::Vec<dim>>& h_centroids,
                thrust::host_vector<kmeans::Vec<dim>>& h_objects
            ) :
                d_centroids(h_centroids),
                d_objects(h_objects),
                d_memberships(h_objects.size()),
                d_deltas(h_objects.size()),

                d_aos_objects(h_objects),
                d_reduced_objects(h_centroids.size()),
                d_reduced_memberships(h_centroids.size()),
                d_reduced_counts(h_centroids.size()),
                d_ones(h_objects.size(), 1)
            {
                this->h_objects = &h_objects;
                this->h_centroids = &h_centroids;
            }

            DeviceDataRaw<dim> to_raw_pointers() {
                DeviceDataRaw<dim> raw_data;
                raw_data.object_count = d_objects.size();
                raw_data.d_objects = d_objects.raw_data();
                raw_data.centroid_count = d_centroids.size();
                raw_data.d_centroids = d_centroids.raw_data();
                raw_data.d_memberships = thrust::raw_pointer_cast(d_memberships.data());
                raw_data.d_deltas = thrust::raw_pointer_cast(d_deltas.data());

                raw_data.d_reduced_objects = thrust::raw_pointer_cast(d_reduced_objects.data());
                raw_data.d_reduced_memberships =
                    thrust::raw_pointer_cast(d_reduced_memberships.data());
                raw_data.d_reduced_counts = thrust::raw_pointer_cast(d_reduced_counts.data());
                return raw_data;
            }

            thrust::host_vector<size_t> get_host_memberships() {
                return static_cast<thrust::host_vector<size_t>>(d_memberships);
            }

            thrust::host_vector<kmeans::Vec<dim>> get_host_centroids() {
                return d_centroids.to_host();
            }
        };
    }
}

#endif
