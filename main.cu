#include <iostream>
#include <fstream>

#include "kmeans_cpu.h"
#include "kmeans_gpu.h"
#include "centroid_init.h"
#include "timers.h"

static constexpr size_t PROG_ARG_PROGRAM_NAME = 0;
static constexpr size_t PROG_ARG_INPUT_FILE = 1;
static constexpr size_t PROG_ARG_METRICS_FILE = 2;
static constexpr size_t MIN_NUM_PROG_ARG = 2;

static constexpr size_t DIMENSION = 3;

static constexpr int VERSION = 1;

static constexpr bool DISABLE_CPU = true;
static constexpr bool SHOW_RESULTS = false;

void usage(char* program_name) {
    std::cout << "USAGE: " << program_name << " input_file [metrics_file]" << std::endl;
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
    if(!SHOW_RESULTS) {
        return;
    }

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

void print_common_metrics(std::ostream& out_stream, size_t object_count, size_t centroid_count) {
    out_stream << object_count << "," << centroid_count << "," << DIMENSION << ",";
    timers::print_common_results(out_stream);
}

void print_metrics(std::ostream& out_stream, size_t object_count, size_t centroid_count) {
    out_stream << std::endl;

    out_stream << "gpu,metod1," << VERSION << ",";
    print_common_metrics(out_stream, object_count, centroid_count);
    timers::gpu::print_results(out_stream);

    if(!DISABLE_CPU) {
        out_stream << std::endl;

        out_stream << "cpu,metod1," << VERSION << ",";
        print_common_metrics(out_stream, object_count, centroid_count);
        timers::cpu::print_results(out_stream);
    }
}

int main(int argc, char* argv[]) {
    if(argc < MIN_NUM_PROG_ARG) {
        usage(argv[PROG_ARG_PROGRAM_NAME]);
    }

    size_t k = 6;

    timers::data_loading.start();
    thrust::host_vector<kmeans::Vec<DIMENSION>> objects =
        load_data(argv[PROG_ARG_INPUT_FILE]);
    timers::data_loading.stop();

    timers::centroid_init.start();
    thrust::host_vector<kmeans::Vec<DIMENSION>> random_centroids =
        kmeans::randomly_init_centroids(k, objects);
    timers::centroid_init.stop();

    thrust::host_vector<kmeans::Vec<DIMENSION>> cpu_centroids = random_centroids;
    thrust::host_vector<kmeans::Vec<DIMENSION>> gpu_centroids = random_centroids;

    std::cout << "Starting GPU computation" << std::endl;
    timers::gpu::algorithm.start();
    auto gpu_memberships = kmeans_gpu::kmeans_clustering(gpu_centroids, objects);
    timers::gpu::algorithm.stop();

    if(!DISABLE_CPU) {
        std::cout << "Starting CPU computation" << std::endl;
        timers::cpu::algorithm.start();
        auto cpu_memberships = kmeans_cpu::kmeans_clustering(cpu_centroids, objects);
        timers::cpu::algorithm.stop();
    }

    std::cout << std::endl;

    std::cout << "GPU results:" << std::endl;
    print_results(std::cout, objects, gpu_centroids, gpu_memberships);
    std::cout << std::endl;

    if(!DISABLE_CPU) {
        std::cout << "CPU results:" << std::endl;
        print_results(std::cout, objects, gpu_centroids, gpu_memberships);
        std::cout << std::endl;
    }

    if(argc == 3) {
        std::ofstream out_file(argv[PROG_ARG_METRICS_FILE], std::ios_base::app);
        print_metrics(out_file, objects.size(), k);
    }
    else {
        print_metrics(std::cout, objects.size(), k);
    }

    // Required, if destructors are allowed to fire automatically cudaErrorCudartUnloading
    // is going to be reported due to driver shutting down.
    timers::destroy_device_timers();
    return 0;
}
