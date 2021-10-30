#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <iostream>
#include <chrono>

namespace kmeans_cpu {
    static constexpr double ACCURACY_THRESHOLD = 0.001;

    template<size_t dim>
    struct Vec {
        double coords[dim];

        Vec() {
            memset(coords, 0, sizeof(double) * dim);
        }

        Vec(double coords[dim]) {
            memcpy(this->coords, coords, sizeof(double) * dim);
        }

        static double distance(Vec<dim>& u, Vec<dim>& v) {
            double square_dist = 0.0;
            for(size_t i = 0; i < dim; ++i) {
                double coord_diff = u.coords[i] - v.coords[i];
                square_dist += coord_diff * coord_diff;
            }

            return std::sqrt(square_dist);
        }
    };

    template <size_t dim>
    std::ostream &operator<<(std::ostream &os, Vec<dim> const &vec) { 
        for(size_t i = 0; i < dim - 1; ++i) {
            os << vec.coords[i];
            os << ", ";
        }

        os << vec.coords[dim - 1];

        return os;
    }

    template<size_t dim>
    std::pair<std::vector<Vec<dim>>, std::vector<size_t>> kmeans_clustering(
        std::vector<Vec<dim>>& objects, int k
    ) {
        double delta = std::numeric_limits<double>::infinity();
        std::vector<Vec<dim>> centroids = initialize_centroids(k, objects);
        std::vector<Vec<dim>> new_centroids(centroids.size());
        std::vector<size_t> new_cluster_size(new_centroids.size());
        std::vector<size_t> memberships(objects.size());

        int it = 0;
        while(delta / objects.size() > ACCURACY_THRESHOLD) {
            delta = 0;
            std::cout << "iteration " << it++ << std::endl;
            calculate_new_centroids(
                objects, centroids, memberships, new_centroids, new_cluster_size, delta
            );
            update_centroids(centroids, new_centroids, new_cluster_size);
        }

        return std::make_pair(centroids, memberships);
    }

    template<size_t dim>
    std::vector<Vec<dim>> initialize_centroids(size_t k, std::vector<Vec<dim>>& objects) {
        std::pair<Vec<dim>, Vec<dim>> bounding_box = calculate_bounding_box(k, objects);
        std::vector<Vec<dim>> centroids(k);
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
    std::pair<Vec<dim>, Vec<dim>> calculate_bounding_box(size_t k, std::vector<Vec<dim>>& objects) {
        Vec<dim> low_bounding_box;
        Vec<dim> high_bounding_box;

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

    template<size_t dim>
    void calculate_new_centroids(
        std::vector<Vec<dim>>& objects,
        std::vector<Vec<dim>>& centroids,
        std::vector<size_t>& memberships,
        std::vector<Vec<dim>>& new_clusters,
        std::vector<size_t>& new_cluster_sizes,
        double& delta
    ) {
        for(size_t i = 0; i < objects.size(); ++i) {
            size_t closest_centroid =
                assign_to_closest_centroid(objects, i, centroids, memberships, delta);

            for(size_t j = 0; j < dim; ++j) {
                new_clusters[closest_centroid].coords[j] += objects[i].coords[j];
            }

            new_cluster_sizes[closest_centroid] += 1;
        }
    }

    template<size_t dim>
    size_t assign_to_closest_centroid(
        std::vector<Vec<dim>>& objects,
        size_t object_idx,
        std::vector<Vec<dim>>& centroids,
        std::vector<size_t>& memberships,
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
    size_t find_closest_centroid(Vec<dim>& object, std::vector<Vec<dim>>& clusters) {
        size_t closest_centroid;
        double min_dist = std::numeric_limits<double>::infinity();
        for(size_t j = 0; j < clusters.size(); ++j) {
            double dist = Vec<dim>::distance(object, clusters[j]);
            if(dist < min_dist) {
                min_dist = dist;
                closest_centroid = j;
            }
        }

        return closest_centroid;
    }

    template<size_t dim>
    void update_centroids(
        std::vector<Vec<dim>>& centroids,
        std::vector<Vec<dim>>& new_centroids,
        std::vector<size_t>& new_cluster_sizes
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
}
