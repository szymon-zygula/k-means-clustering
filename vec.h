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
}

#endif
