#include <iostream>
#include <fstream>

#include "kmeans_cpu.cuh"
#include "kmeans_gpu_method1.cuh"
#include "kmeans_gpu_method2.cuh"
#include "centroid_init.cuh"
#include "timers.cuh"
#include "config.cuh"

static constexpr size_t PROG_ARG_PROGRAM_NAME = 0;
static constexpr size_t PROG_ARG_INPUT_FILE = 1;
static constexpr size_t PROG_ARG_CENTROID_COUNT = 2;
static constexpr size_t PROG_ARG_METRICS_FILE = 3;
static constexpr size_t MIN_NUM_PROG_ARG = 3;

void usage(char* program_name) {
    std::cout << "USAGE: " << program_name <<
        " input_file centroid_count [metrics_file]" << std::endl;
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
    thrust::host_vector<size_t>& memberships,
    std::string who
) {
    if(!SHOW_RESULTS) {
        return;
    }

    out_stream << std::endl;
    out_stream << who << " results:" << std::endl;
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

void print_cpu_metrics(std::ostream& out_stream, size_t object_count, size_t centroid_count) {
    out_stream << "cpu,method1," << VERSION << ",";
    print_common_metrics(out_stream, object_count, centroid_count);
    timers::cpu::print_results(out_stream);
    out_stream << std::endl;
}

void print_gpu_metrics(
    std::ostream& out_stream,
    size_t object_count,
    size_t centroid_count,
    size_t method
) {
    out_stream << "gpu,method" << method << "," << VERSION << ",";
    print_common_metrics(out_stream, object_count, centroid_count);
    timers::gpu::print_results(out_stream);
    out_stream << std::endl;
}

int main(int argc, char* argv[]) {
    if(argc < MIN_NUM_PROG_ARG) {
        usage(argv[PROG_ARG_PROGRAM_NAME]);
    }

    size_t centroid_count = atoi(argv[PROG_ARG_CENTROID_COUNT]);
    if(centroid_count == 0) {
        usage(argv[PROG_ARG_PROGRAM_NAME]);
    }

    timers::data_loading.start();
    thrust::host_vector<kmeans::Vec<DIMENSION>> objects =
        load_data(argv[PROG_ARG_INPUT_FILE]);
    timers::data_loading.stop();

    timers::centroid_init.start();
    thrust::host_vector<kmeans::Vec<DIMENSION>> random_centroids =
        kmeans::randomly_init_centroids(centroid_count, objects);
    timers::centroid_init.stop();

    thrust::host_vector<kmeans::Vec<DIMENSION>> gpu1_objects = objects;
    thrust::host_vector<kmeans::Vec<DIMENSION>> gpu2_objects = objects;

    thrust::host_vector<kmeans::Vec<DIMENSION>> gpu1_centroids = random_centroids;
    thrust::host_vector<kmeans::Vec<DIMENSION>> gpu2_centroids = random_centroids;

    std::ofstream metrics_out_file;
    std::ostream* metrics_out_stream;
    if(argc == 3) {
        metrics_out_file.open(argv[PROG_ARG_METRICS_FILE], std::ios_base::app);
        metrics_out_stream = &metrics_out_file;
    }
    else {
        metrics_out_stream = &std::cout;
    }

    if(!DISABLE_CPU) {
        thrust::host_vector<kmeans::Vec<DIMENSION>> cpu_centroids = random_centroids;

        std::cerr << "Starting CPU computation" << std::endl;
        timers::cpu::algorithm.start();
        auto cpu_memberships = kmeans_cpu::kmeans_clustering(cpu_centroids, objects);
        timers::cpu::algorithm.stop();

        print_results(std::cout, objects, cpu_centroids, cpu_memberships, "CPU");
        print_cpu_metrics(*metrics_out_stream, objects.size(), centroid_count);
    }

    {
        thrust::host_vector<kmeans::Vec<DIMENSION>> gpu1_centroids = random_centroids;

        std::cerr << "Starting gpu1 computation" << std::endl;
        timers::gpu::algorithm.start();
        auto gpu1_memberships =
            kmeans_gpu::method1::kmeans_clustering(gpu1_centroids, objects);
        timers::gpu::algorithm.stop();

        print_results(std::cout, objects, gpu1_centroids, gpu1_memberships, "gpu1");
        print_gpu_metrics(*metrics_out_stream, objects.size(), centroid_count, 1);
    }

    {
        thrust::host_vector<kmeans::Vec<DIMENSION>> gpu2_centroids = random_centroids;
        timers::reset_gpu_timers();

        std::cerr << "Starting gpu2 computation" << std::endl;
        timers::gpu::algorithm.start();
        auto gpu2_memberships =
            kmeans_gpu::method2::kmeans_clustering(gpu2_centroids, objects);
        timers::gpu::algorithm.stop();

        print_results(std::cout, objects, gpu2_centroids, gpu2_memberships, "gpu2");
        print_gpu_metrics(*metrics_out_stream, objects.size(), centroid_count, 2);
    }

    // Required, if destructors are allowed to fire automatically cudaErrorCudartUnloading
    // is going to be reported due to driver shutting down.
    timers::destroy_device_timers();
    return 0;
}
