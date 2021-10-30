#ifndef KMEANS_CLUSTERING_CPU_H
#define KMEANS_CLUSTERING_CPU_H

#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <iostream>
#include <chrono>

#include <thrust/host_vector.h>

#include "kmeans_clustering_common.h"

namespace kmeans_cpu {
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

    // TODO: Unify with GPU code, allow to be chosen before kmeans_clustering
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
    std::pair<thrust::host_vector<kmeans::Vec<dim>>, thrust::host_vector<size_t>> kmeans_clustering(
        thrust::host_vector<kmeans::Vec<dim>>& objects, int k
    ) {
        double delta = std::numeric_limits<double>::infinity();
        thrust::host_vector<kmeans::Vec<dim>> centroids = initialize_centroids(k, objects);
        thrust::host_vector<kmeans::Vec<dim>> new_centroids(centroids.size());
        thrust::host_vector<size_t> new_cluster_size(new_centroids.size());
        thrust::host_vector<size_t> memberships(objects.size());

        while(delta / objects.size() > kmeans::ACCURACY_THRESHOLD) {
            delta = 0;
            calculate_new_centroids(
                objects, centroids, memberships, new_centroids, new_cluster_size, delta
            );
            update_centroids(centroids, new_centroids, new_cluster_size);
        }

        return std::make_pair(centroids, memberships);
    }
}

#endif
