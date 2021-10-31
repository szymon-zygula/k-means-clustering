#ifndef KMEANS_CLUSTERING_COMMON_H
#define KMEANS_CLUSTERING_COMMON_H

#include <cmath>
#include <ostream>
#include <random>
#include <chrono>

#include <thrust/host_vector.h>

namespace kmeans {
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

        static double square_distance(Vec<dim>& u, Vec<dim>& v) {
            double square_dist = 0.0;
            for(size_t i = 0; i < dim; ++i) {
                double coord_diff = u.coords[i] - v.coords[i];
                square_dist += coord_diff * coord_diff;
            }

            return square_dist;
        }
    };

    template <size_t dim>
    std::ostream &operator<<(std::ostream &os, kmeans::Vec<dim> const &vec) { 
        for(size_t i = 0; i < dim - 1; ++i) {
            os << vec.coords[i];
            os << ", ";
        }

        os << vec.coords[dim - 1];

        return os;
    }

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

}

#endif
