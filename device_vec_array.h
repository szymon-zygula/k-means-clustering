#ifndef DEVICE_VEC_ARRAY_H
#define DEVICE_VEC_ARRAY_H

#include "vec.h"

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
            _size = objects.size();
            thrust::host_vector<double> rearranged_objects = objects_aos_to_soa(objects);
            init_from_rearranged_objects(rearranged_objects);
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
            thrust::host_vector<double> objects(_size * dim);
            cudaMemcpy(
                objects.data(), d_array,
                sizeof(double) * _size * dim,
                cudaMemcpyDeviceToHost
            );

            return objects_soa_to_aos(objects);
        }

        ~DeviceVecArray() {
            cudaFree(d_array);
        }

        private:
        thrust::host_vector<double> objects_aos_to_soa(
            thrust::host_vector<kmeans::Vec<dim>>& objects
        ) {
            thrust::host_vector<double> rearranged_objects(_size * dim);

            for(size_t i = 0; i < dim; ++i) {
                for(size_t j = 0; j < _size; ++j) {
                    rearranged_objects[j + i * _size] = objects[j].coords[i];
                }
            }

            return rearranged_objects;
        }

        thrust::host_vector<kmeans::Vec<dim>> objects_soa_to_aos(
            thrust::host_vector<double> objects
        ) {
            thrust::host_vector<kmeans::Vec<dim>> rearranged_objects(_size);

            for(size_t i = 0; i < dim; ++i) {
                for(size_t j = 0; j < _size; ++j) {
                    rearranged_objects[j].coords[i] = objects[j + i * _size];
                }
            }

            return rearranged_objects;
        }

        void init_from_rearranged_objects(thrust::host_vector<double>& rearranged_objects) {
            cudaMalloc(&d_array, sizeof(double) * _size * dim);
            cudaMemcpy(
                d_array, rearranged_objects.data(),
                sizeof(double) * _size * dim,
                cudaMemcpyHostToDevice
            );
        }
    };
}

#endif
