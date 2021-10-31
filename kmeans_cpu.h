#ifndef KMEANS_CLUSTERING_CPU_H
#define KMEANS_CLUSTERING_CPU_H

#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <iostream>

#include <thrust/host_vector.h>

#include "vec.h"
#include "timers.h"

namespace kmeans_cpu {
    template<size_t dim>
    size_t find_closest_centroid(
        kmeans::Vec<dim>& object,
        thrust::host_vector<kmeans::Vec<dim>>& clusters
    ) {
        size_t closest_centroid;
        double min_dist = std::numeric_limits<double>::infinity();
        for(size_t i = 0; i < clusters.size(); ++i) {
            double dist = kmeans::Vec<dim>::square_distance(object, clusters[i]);
            if(dist < min_dist) {
                min_dist = dist;
                closest_centroid = i;
            }
        }

        return closest_centroid;
    }

    template<size_t dim>
    size_t assign_to_closest_centroid(
        thrust::host_vector<kmeans::Vec<dim>>& objects,
        size_t object_idx,
        thrust::host_vector<kmeans::Vec<dim>>& centroids,
        thrust::host_vector<size_t>& memberships,
        double& delta
    ) {
        size_t closest_centroid = find_closest_centroid(objects[object_idx], centroids);
        if(memberships[object_idx] != closest_centroid) {
            delta += 1.0;
            memberships[object_idx] = closest_centroid;
        }

        return closest_centroid;
    }

    template<size_t dim>
    void calculate_new_centroids(
        thrust::host_vector<kmeans::Vec<dim>>& objects,
        thrust::host_vector<kmeans::Vec<dim>>& centroids,
        thrust::host_vector<size_t>& memberships,
        thrust::host_vector<kmeans::Vec<dim>>& new_centroids,
        thrust::host_vector<size_t>& new_cluster_sizes,
        double& delta
    ) {
        for(size_t i = 0; i < objects.size(); ++i) {
            size_t closest_centroid =
                assign_to_closest_centroid(objects, i, centroids, memberships, delta);

            for(size_t j = 0; j < dim; ++j) {
                new_centroids[closest_centroid].coords[j] += objects[i].coords[j];
            }   

            new_cluster_sizes[closest_centroid] += 1;
        }
    }

    template<size_t dim>
    void update_centroids(
        thrust::host_vector<kmeans::Vec<dim>>& centroids,
        thrust::host_vector<kmeans::Vec<dim>>& new_centroids,
        thrust::host_vector<size_t>& new_cluster_sizes
    ) {
        for(size_t i = 0; i < centroids.size(); ++i) {
            for(size_t j = 0; j < dim; ++j) {
                centroids[i].coords[j] =
                    new_centroids[i].coords[j] / std::max(new_cluster_sizes[i], (size_t)1);
                new_centroids[i].coords[j] = 0.0;
            }

            new_cluster_sizes[i] = 0;
        }
    }

    template<size_t dim>
    thrust::host_vector<size_t> kmeans_clustering(
        thrust::host_vector<kmeans::Vec<dim>>& centroids,
        thrust::host_vector<kmeans::Vec<dim>>& objects
    ) {
        double delta = std::numeric_limits<double>::infinity();
        thrust::host_vector<kmeans::Vec<dim>> new_centroids(centroids.size());
        thrust::host_vector<size_t> new_cluster_size(new_centroids.size());
        thrust::host_vector<size_t> memberships(objects.size());

        while(delta / objects.size() > kmeans::ACCURACY_THRESHOLD) {
            delta = 0;
            timers::cpu::distance_calculation.start();
            calculate_new_centroids(
                objects, centroids, memberships, new_centroids, new_cluster_size, delta
            );
            timers::cpu::distance_calculation.stop();

            timers::cpu::new_centroid_calculation.start();
            update_centroids(centroids, new_centroids, new_cluster_size);
            timers::cpu::new_centroid_calculation.stop();
        }

        return memberships;
    }
}

#endif
