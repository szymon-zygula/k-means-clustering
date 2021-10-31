#ifndef CENTROID_INIT_H
#define CENTROID_INIT_H

#include "vec.h"

namespace kmeans {
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
    thrust::host_vector<kmeans::Vec<dim>> randomly_init_centroids(
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
