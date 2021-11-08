#ifndef TIMERS_H
#define TIMERS_H

#include <chrono>
#include <ostream>

namespace timers {
    class Timer {
        public:
        virtual void start() = 0;
        virtual void stop() = 0;
        virtual float elapsed_ms() = 0;
        virtual void reset() = 0;
        virtual ~Timer() {};
    };

    class HostTimer : public Timer {
        std::chrono::steady_clock::time_point start_point;
        std::chrono::microseconds total_elapsed;

        public:
        void start();
        void stop();
        float elapsed_ms();
        void reset();
        ~HostTimer();
    };

    class DeviceTimer : public Timer {
        float total_milliseconds_elapsed = 0.0f;
        float milliseconds_elapsed;
        cudaEvent_t start_ev;
        cudaEvent_t stop_ev;

        size_t times_started = 0;
        bool total_elapsed_is_updated = false;

        public:
        void start();
        void stop();
        float elapsed_ms();
        void reset();

        ~DeviceTimer();

        private:
        void init_events();
        void update_total_elapsed_time();
        void destroy_events();
    };

    void print_common_results(std::ostream& stream);

    extern HostTimer data_loading;
    extern HostTimer centroid_init;

    namespace cpu {
        void print_results(std::ostream& stream);

        extern HostTimer algorithm;
        extern HostTimer distance_calculation;
        extern HostTimer new_centroid_calculation;
    }

    namespace gpu {
        void print_results(std::ostream& stream);

        extern DeviceTimer aos_to_soa_conversion;
        extern DeviceTimer host_to_device_transfer;

        extern HostTimer algorithm;
        extern DeviceTimer distance_calculation;
        extern DeviceTimer new_centroid_calculation;

        extern DeviceTimer device_to_host_transfer;
        extern DeviceTimer soa_to_aos_conversion;
    }

    void reset_gpu_timers();
    void destroy_device_timers();
}

#endif
