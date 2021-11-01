#include "timers.h"

namespace timers {
    void HostTimer::start() {
        start_point = std::chrono::steady_clock::now();
    }

    void HostTimer::stop() {
        auto stop_point = std::chrono::steady_clock::now();
        total_elapsed += std::chrono::duration_cast<std::chrono::microseconds>(
            stop_point - start_point
        );
    }

    float HostTimer::elapsed_ms() {
        return total_elapsed.count() / 1000.0f;
    }

    void HostTimer::reset() {
        start_point = std::chrono::steady_clock::time_point();
        total_elapsed = std::chrono::microseconds();
    }

    HostTimer::~HostTimer() {
        // Do nothing
    }

    void DeviceTimer::start() {
        if(times_started > 0) {
            update_total_elapsed_time();
            destroy_events();
        }

        init_events();
        times_started += 1;

        cudaEventRecord(start_ev);
    }

    void DeviceTimer::stop() {
        cudaEventRecord(stop_ev);
        total_elapsed_is_updated = false;
    }

    float DeviceTimer::elapsed_ms() {
        if(!total_elapsed_is_updated) {
            update_total_elapsed_time();
            total_elapsed_is_updated = true;
        }

        return total_milliseconds_elapsed;
    }

    DeviceTimer::~DeviceTimer() {
        if(times_started > 0) {
            times_started = 0;
            destroy_events();
        }
    }

    void DeviceTimer::init_events() {
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);
    }

    void DeviceTimer::update_total_elapsed_time() {
        cudaEventElapsedTime(&milliseconds_elapsed, start_ev, stop_ev);
        total_milliseconds_elapsed += milliseconds_elapsed;
    }

    void DeviceTimer::destroy_events() {
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }

    void DeviceTimer::reset() {
        total_milliseconds_elapsed = 0.0f;
        if(times_started > 0) {
            destroy_events();
            times_started = 0;
            total_elapsed_is_updated = false;
        }
    }

    void print_common_results(std::ostream& stream) {
        stream
            << data_loading.elapsed_ms() << ","
            << centroid_init.elapsed_ms() << ",";
    }

    void gpu::print_results(std::ostream& stream) {
        stream
            << aos_to_soa_conversion.elapsed_ms() << ","
            << host_to_device_transfer.elapsed_ms() << ","
            << algorithm.elapsed_ms() << ","
            << distance_calculation.elapsed_ms() << ","
            << new_centroid_calculation.elapsed_ms() << ","
            << device_to_host_transfer.elapsed_ms() << ","
            << soa_to_aos_conversion.elapsed_ms() << ",";
    }

    void cpu::print_results(std::ostream& stream) {
        stream
            << 0.0f << ","
            << 0.0f << ","
            << algorithm.elapsed_ms() << ","
            << distance_calculation.elapsed_ms() << ","
            << new_centroid_calculation.elapsed_ms() << ","
            << 0.0f << ","
            << 0.0f << ",";
    }

    HostTimer data_loading;
    HostTimer centroid_init;

    namespace cpu {
        void print_results(std::ostream& stream);

        HostTimer algorithm;
        HostTimer distance_calculation;
        HostTimer new_centroid_calculation;
    }

    namespace gpu {
        void print_results(std::ostream& stream);

        HostTimer aos_to_soa_conversion;
        DeviceTimer host_to_device_transfer;

        HostTimer algorithm;
        DeviceTimer distance_calculation;
        DeviceTimer new_centroid_calculation;

        DeviceTimer device_to_host_transfer;
        HostTimer soa_to_aos_conversion;
    }

    void reset_gpu_timers() {
        gpu::aos_to_soa_conversion.reset();
        gpu::host_to_device_transfer.reset();

        gpu::algorithm.reset();
        gpu::distance_calculation.reset();
        gpu::new_centroid_calculation.reset();

        gpu::device_to_host_transfer.reset();
        gpu::soa_to_aos_conversion.reset();
    }

    void destroy_device_timers() {
        gpu::host_to_device_transfer.~DeviceTimer();
        gpu::distance_calculation.~DeviceTimer();
        gpu::new_centroid_calculation.~DeviceTimer();
        gpu::device_to_host_transfer.~DeviceTimer();
    }
}
