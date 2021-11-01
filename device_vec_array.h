#ifndef DEVICE_VEC_ARRAY_H
#define DEVICE_VEC_ARRAY_H

#include "vec.h"
#include "timers.h"

namespace kmeans_gpu {
    template<size_t dim>
    class DeviceVecArray {
        size_t _size;
        double* d_array;

        public:
        DeviceVecArray(size_t size) {
            _size = size;
            cudaMalloc(&d_array, sizeof(double) * _size * dim);
            cudaMemset(d_array, 0, sizeof(double) * _size * dim);
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
            timers::gpu::device_to_host_transfer.start();
            thrust::host_vector<double> objects(_size * dim);
            cudaMemcpy(
                objects.data(), d_array,
                sizeof(double) * _size * dim,
                cudaMemcpyDeviceToHost
            );

            timers::gpu::device_to_host_transfer.stop();
            return objects_soa_to_aos(objects);
        }

        ~DeviceVecArray() {
            cudaFree(d_array);
        }

        private:
        void init_from_host_vector(thrust::host_vector<kmeans::Vec<dim>>& objects) {
            _size = objects.size();
            thrust::host_vector<double> rearranged_objects = objects_aos_to_soa(objects);
            init_from_rearranged_objects(rearranged_objects);
        }

        thrust::host_vector<double> objects_aos_to_soa(
            thrust::host_vector<kmeans::Vec<dim>>& objects
        ) {
            timers::gpu::aos_to_soa_conversion.start();
            thrust::host_vector<double> rearranged_objects(_size * dim);

            for(size_t i = 0; i < dim; ++i) {
                for(size_t j = 0; j < _size; ++j) {
                    rearranged_objects[j + i * _size] = objects[j].coords[i];
                }
            }

            timers::gpu::aos_to_soa_conversion.stop();
            return rearranged_objects;
        }

        thrust::host_vector<kmeans::Vec<dim>> objects_soa_to_aos(
            thrust::host_vector<double> objects
        ) {
            timers::gpu::soa_to_aos_conversion.start();
            thrust::host_vector<kmeans::Vec<dim>> rearranged_objects(_size);

            for(size_t i = 0; i < dim; ++i) {
                for(size_t j = 0; j < _size; ++j) {
                    rearranged_objects[j].coords[i] = objects[j + i * _size];
                }
            }

            timers::gpu::soa_to_aos_conversion.stop();
            return rearranged_objects;
        }

        void init_from_rearranged_objects(thrust::host_vector<double>& rearranged_objects) {
            timers::gpu::host_to_device_transfer.start();
            cudaMalloc(&d_array, sizeof(double) * _size * dim);
            cudaMemcpy(
                d_array, rearranged_objects.data(),
                sizeof(double) * _size * dim,
                cudaMemcpyHostToDevice
            );
            timers::gpu::host_to_device_transfer.stop();
        }
    };
}

#endif
