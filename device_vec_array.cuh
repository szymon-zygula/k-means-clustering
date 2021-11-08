#ifndef DEVICE_VEC_ARRAY_H
#define DEVICE_VEC_ARRAY_H

#include "config.cuh"

#include "vec.cuh"
#include "timers.cuh"

namespace kmeans_gpu {
    template<size_t dim>
    __global__
    void aos_to_soa_vectors(double* d_aos, double* d_soa, size_t len) {
        size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid >= len) {
            return;
        }

        for(size_t i = 0; i < dim; ++i) {
            d_soa[tid + i * len] = d_aos[tid * dim + i];
        }
    }

    template<size_t dim>
    __global__
    void soa_to_aos_vectors(double* d_soa, double* d_aos, size_t len) {
        size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid >= len) {
            return;
        }

        for(size_t i = 0; i < dim; ++i) {
            d_aos[tid * dim + i] = d_soa[tid + i * len];
        }
    }

    template<size_t dim>
    class DeviceVecArray {
        size_t _size;
        double* d_array;

        public:
        DeviceVecArray(size_t size) {
            _size = size;
            cudaMalloc(&d_array, sizeof(kmeans::Vec<dim>) * _size);
            cudaMemset(d_array, 0, sizeof(kmeans::Vec<dim>) * _size);
        }

        DeviceVecArray(thrust::host_vector<kmeans::Vec<dim>>& objects) {
            init_from_host_vector(objects);
        }

        void operator=(thrust::host_vector<kmeans::Vec<dim>>& objects) {
            init_from_host_vector(objects);
        }

        size_t size() {
            return _size;
        }

        double* raw_data() {
            return d_array;
        }

        __device__
        static double& get(double* d_array, size_t size, size_t vec_idx, size_t dim_idx) {
            return d_array[vec_idx + dim_idx * size];
        }

        thrust::host_vector<kmeans::Vec<dim>> to_host() {
            double* d_aos;
            cudaMalloc(&d_aos, sizeof(kmeans::Vec<dim>) * _size);

            size_t block_count =
                (_size + THREADS_PER_BLOCK - 1) /
                THREADS_PER_BLOCK;

            timers::gpu::soa_to_aos_conversion.start();
            soa_to_aos_vectors<dim><<<
                block_count, THREADS_PER_BLOCK
            >>>(
                d_array, d_aos, _size
            );
            timers::gpu::soa_to_aos_conversion.stop();

            thrust::host_vector<kmeans::Vec<dim>> objects(_size);
            timers::gpu::device_to_host_transfer.start();
            cudaMemcpy(
                objects.data(), d_aos,
                sizeof(kmeans::Vec<dim>) * _size,
                cudaMemcpyDeviceToHost
            );
            timers::gpu::device_to_host_transfer.stop();

            cudaFree(d_aos);

            return objects;
        }

        ~DeviceVecArray() {
            cudaFree(d_array);
        }

        private:
        void init_from_host_vector(thrust::host_vector<kmeans::Vec<dim>>& objects) {
            _size = objects.size();
            double* d_aos;

            cudaMalloc(&d_array, sizeof(kmeans::Vec<dim>) * _size);
            cudaMalloc(&d_aos, sizeof(kmeans::Vec<dim>) * _size);

            timers::gpu::host_to_device_transfer.start();
            cudaMemcpy(
                d_aos, objects.data(),
                sizeof(kmeans::Vec<dim>) * _size,
                cudaMemcpyHostToDevice
            );
            timers::gpu::host_to_device_transfer.stop();

            size_t block_count =
                (_size + THREADS_PER_BLOCK - 1) /
                THREADS_PER_BLOCK;

            timers::gpu::aos_to_soa_conversion.start();
            aos_to_soa_vectors<dim><<<
                block_count, THREADS_PER_BLOCK
            >>>(
                d_aos, d_array, _size
            );
            timers::gpu::aos_to_soa_conversion.stop();

            cudaFree(d_aos);
        }
    };
}

#endif
