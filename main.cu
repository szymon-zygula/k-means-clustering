#include <iostream>
#include <fstream>

#include "kmeans_clustering_cpu.h"
#include "kmeans_clustering_gpu.h"

static constexpr size_t PROG_ARG_PROGRAM_NAME = 0;
static constexpr size_t PROG_ARG_INPUT_FILE = 1;
static constexpr size_t NUM_PROG_ARG = 2;

static constexpr size_t DIMENSION = 2;

void usage(char* program_name) {
    std::cout << "USAGE: " << program_name << " input_file" << std::endl;
    exit(1);
}

thrust::host_vector<kmeans::Vec<DIMENSION>> load_data(char* input_file_path) {
    thrust::host_vector<kmeans::Vec<DIMENSION>> objects;

    std::ifstream input_file(input_file_path);
    bool reading = true;
    while(reading) {
        objects.resize(objects.size() + 1);
        for(size_t i = 0; i < DIMENSION; ++i) {
            input_file >> objects.back().coords[i];
            if(!input_file) {
                reading = false;
                break;
            }
        }
    }
    objects.resize(objects.size() - 1);

    return objects;
}

void print_results(
    std::ostream& out_stream,
    thrust::host_vector<kmeans::Vec<DIMENSION>>& objects,
    thrust::host_vector<kmeans::Vec<DIMENSION>>& centroids,
    thrust::host_vector<size_t>& memberships
) {
    for(size_t i = 0; i < centroids.size(); ++i) {
        out_stream << std::endl;
        out_stream << "Centroid: [" << centroids[i] << "]" << std::endl;
        for(size_t j = 0; j < objects.size(); ++j) {
            if(memberships[j] == i) {
                out_stream << "[" << objects[j] << "]" << std::endl;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if(argc != NUM_PROG_ARG) {
        usage(argv[PROG_ARG_PROGRAM_NAME]);
    }

    size_t k = 6;

    thrust::host_vector<kmeans::Vec<DIMENSION>> objects =
        load_data(argv[PROG_ARG_INPUT_FILE]);
    thrust::host_vector<kmeans::Vec<DIMENSION>> centroids =
        kmeans::initialize_centroids(k, objects);
    auto memberships = kmeans_gpu::kmeans_clustering(centroids, objects);

    print_results(std::cout, objects, centroids, memberships);

    return 0;
}
