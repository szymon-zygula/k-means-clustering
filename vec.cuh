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

        __host__ __device__
        Vec() {
            for(size_t i = 0; i < dim; ++i) {
                coords[i] = 0.0;
            }
        }

        static double square_distance(Vec<dim>& u, Vec<dim>& v) {
            double square_dist = 0.0;
            for(size_t i = 0; i < dim; ++i) {
                double coord_diff = u.coords[i] - v.coords[i];
                square_dist += coord_diff * coord_diff;
            }

            return square_dist;
        }

        __host__ __device__
        Vec operator+(const Vec& vec) const {
            Vec<dim> sum(coords);
            for(size_t i = 0; i < dim; ++i) {
                sum.coords[i] += vec.coords[i];
            }

            return sum;
        }

        private:
        __host__ __device__
        Vec(const double coords[dim]) {
            for(size_t i = 0; i < dim; ++i) {
                this->coords[i] = coords[i];
            }
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
