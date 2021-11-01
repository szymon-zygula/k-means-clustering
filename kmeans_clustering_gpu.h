// This file is included in kmeans_gpu_method(1,2) as the definition of this function
// would be the same in both methods (except for namespaces).

template<size_t dim>
thrust::host_vector<size_t> kmeans_clustering(
    thrust::host_vector<kmeans::Vec<dim>>& h_centroids,
    thrust::host_vector<kmeans::Vec<dim>>& h_objects
) {
    double delta = std::numeric_limits<double>::infinity();
    DeviceData<dim> data(h_centroids, h_objects);

    while(delta / h_objects.size() > kmeans::ACCURACY_THRESHOLD) {
        timers::gpu::distance_calculation.start();
        delta = calculate_nearest_centroids(data);
        timers::gpu::distance_calculation.stop();

        timers::gpu::new_centroid_calculation.start();
        update_centroids(data);
        timers::gpu::new_centroid_calculation.stop();
    }

    h_centroids = data.get_host_centroids();
    h_objects = data.d_objects.to_host();
    return data.get_host_memberships();
}
