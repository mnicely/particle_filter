/**
 * @file particle_bpf_gpu_impl.h
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains parallel implementation of bootstrap
 * particle filter.
 *
 * @copyright Copyright (c) 2020
 *
 * @license This project is released under the GNU Public License
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */

#include <algorithm>  // std::transform, std::remove_if, std::nth_element
#include <cmath>      // std::sqrt
#include <cstring>    // std::memcpy
#include <iterator>   // std::vector<>::const_iterator
#include <numeric>    // std::accumulate, std::inner_product

#include <helper_cuda.h>  // checkCudaErrors

#include "models.h"

namespace filters {

/**
 * @brief Wrapper to CUDA kernel InitialFilter
 *
 * @tparam T Data type should be float
 * @param[in] sm_count_ Number of streaming multiprocessors
 * @param[in] cuda_streams_ CUDA streams
 * @param[in] kNumParticles Number of particles
 * @param[in] kSqInitialNoiseCov Squared initial noise covariance data
 * @param[in/out] cuda_events_ CUDA events
 * @param[out] d_particle_state_new_ Latest particle state data
 */
template<typename T>
void InitializeFilterCuda( int const &         sm_count_,
                           cudaStream_t const *cuda_streams_,
                           int const &         kNumParticles,
                           T const *           h_pin_sq_initial_noise_cov_,
                           cudaEvent_t *       cuda_events_,
                           T *                 d_particle_state_new_ );

/**
 * @brief Wrapper to CUDA kernel computeMeasErrors
 *
 * @tparam T Data type should be float
 * @param[in] sm_count_ Number of streaming multiprocessors
 * @param[in] cuda_streams_ CUDA streams
 * @param[in] kNumParticles  Number of particles
 * @param[in] kInvMeasNoiseCov Inverse measurement noise covariance data
 * @param[in] h_pin_meas_update_ Current measurement data
 * @param[in] d_particle_state_new_ Latest particle state data
 * @param[in/out] cuda_events_  CUDA events
 * @param[out] d_particle_weights_ Updated particle weight data
 * @param[out] d_particle_state_ Updated particle state data
 */
template<typename T>
void ComputeMeasErrorsCuda( int const &         sm_count_,
                            cudaStream_t const *cuda_streams_,
                            int const &         kNumParticles,
                            T const *           h_pin_inv_meas_noise_cov_,
                            T const *           h_pin_meas_update_,
                            T const *           d_particle_state_new_,
                            cudaEvent_t *       cuda_events_,
                            T *                 d_particle_weights_,
                            T *                 d_particle_state_ );

/**
 * @brief Wrapper to CUDA kernel ComputeEstimates
 *
 * @tparam T Data type should be float
 * @param[in] sm_count_ Number of streaming multiprocessors
 * @param[in] cuda_streams_ CUDA streams
 * @param[in] kNumParticles Number of particles
 * @param[in] time_step Current time step
 * @param[in] kResamplingMethod Resampling method type
 * @param[in] d_particle_state_ Particle state data
 * @param[in/out] cuda_events_ CUDA events
 * @param[out] d_filtered_estimates_ Updated filtered state estimates
 * @param[out] d_particle_weights_ Updated particle weight data
 */
template<typename T>
void ComputeEstimatesCuda( int const &         sm_count_,
                           cudaStream_t const *cuda_streams_,
                           int const &         kNumParticles,
                           int const &         time_step,
                           int const &         kResamplingMethod,
                           T const *           d_particle_state_,
                           cudaEvent_t *       cuda_events_,
                           T *                 d_filtered_estimates_,
                           T *                 d_particle_weights_ );

/**
 * @brief Wrapper to CUDA kernel ComputeResampleIndex
 *
 * @tparam T Data type should be float
 * @param[in] sm_count_ Number of streaming multiprocessors
 * @param[in] cuda_streams_ CUDA streams
 * @param[in] kNumParticles Number of particles
 * @param[in] time_step Current time step
 * @param[in] kResamplingMethod Resampling method type
 * @param[in] d_particle_weights_ Particle weight data
 * @param[in] d_prefix_sum_particle_weights_ Cumulative summation of particle
 * weights
 * @param[in/out] cuda_events_ CUDA events
 * @param[out] d_resampling_index_up_ Updated resampling index (marching up) for
 * parallel resampling
 * @param[out] d_resampling_index_down_ Updated resampling index (marching down)
 * for parallel resampling
 */
template<typename T>
void ComputeResampleIndexCuda( int const &         sm_count_,
                               cudaStream_t const *cuda_streams_,
                               int const &         kNumParticles,
                               int const &         time_step,
                               int const &         kResamplingMethod,
                               T const *           d_particle_weights_,
                               T *          d_prefix_sum_particle_weights_,
                               cudaEvent_t *cuda_events_,
                               int *        d_resampling_index_up_,
                               int *        d_resampling_index_down_ );

/**
 * @brief Wrapper to CUDA kernel ComputeParticleTransition
 *
 * @tparam T Data type should be float
 * @param[in] sm_count_ Number of streaming multiprocessors
 * @param[in] cuda_streams_ CUDA streams
 * @param[in] kNumParticles Number of particles
 * @param[in] kResamplingMethod Resampling method type
 * @param[in] kSqProcessNoiseCov Squared process noise covariance data
 * @param[in] d_particle_state_ Particle state data
 * @param[in] d_resampling_index_up_ Resampling index (marching up) for parallel
 * resampling
 * @param[in] d_resampling_index_down_ Resampling index (marching down) for
 * parallel resampling
 * @param[in/out] cuda_events_ CUDA events
 * @param[out] d_particle_state_new_ Updated particle state data
 */
template<typename T>
void ComputeParticleTransitionCuda( int const &         sm_count_,
                                    cudaStream_t const *cuda_streams_,
                                    int const &         kNumParticles,
                                    int const &         kResamplingMethod,
                                    T const *    h_pin_sq_process_noise_cov_,
                                    T const *    d_particle_state_,
                                    int const *  d_resampling_index_up_,
                                    int const *  d_resampling_index_down_,
                                    cudaEvent_t *cuda_events_,
                                    T *          d_particle_state_new_ );

template<typename T>
ParticleBpfGpu<T>::ParticleBpfGpu(
    utility::FilterInfo const &filter_info,
    std::function<void( int const &idx, float *current_meas_data )> const
        &truth_meas_func_ptr ) :
    // clang-format off
    device_id_ {},
    sm_count_ {},
    kSysDim { utility::kSysDim }, 
    kMeasDim { utility::kMeasDim },
    kSamples { filter_info.samples }, 
    kNumParticles { filter_info.particles }, 
    kResamplingMethod { filter_info.resampling }, 
    filtered_estimates_( kSysDim * filter_info.samples, 0.0f ),
    truth_meas_func_ptr_ { truth_meas_func_ptr },
    cuda_streams_ {},
    cuda_events_ {},
    d_filtered_estimates_ { DeviceAllocate<T>( kSamples * kSysDim ) }, 
    d_particle_state_ { DeviceAllocate<T>( kNumParticles * kSysDim ) },
    d_particle_state_new_ { DeviceAllocate<T>( kNumParticles * kSysDim ) }, 
    d_particle_weights_ { DeviceAllocate<T>( kNumParticles ) },
    d_prefix_sum_particle_weights_ { DeviceAllocate<T>( kNumParticles ) }, 
    d_resampling_index_up_ { DeviceAllocate<int>( kNumParticles ) }, 
    d_resampling_index_down_ { DeviceAllocate<int>( kNumParticles ) },
    h_pin_sq_initial_noise_cov_ { HostAllocate<T> ( kSysDim * kSysDim ) },
    h_pin_inv_meas_noise_cov_ { HostAllocate<T> ( kMeasDim * kMeasDim ) },
    h_pin_sq_process_noise_cov_ { HostAllocate<T> ( kSysDim * kSysDim ) },
    h_pin_meas_update_ { HostAllocate<T> ( kMeasDim ) }  // clang-format on
{
    // Get number of SMs
    checkCudaErrors( cudaDeviceGetAttribute(
        &sm_count_, cudaDevAttrMultiProcessorCount, device_id_ ) );

    // Create streams and priorities
    int priority_high {}, priority_low {};
    checkCudaErrors(
        cudaDeviceGetStreamPriorityRange( &priority_low, &priority_high ) );

    checkCudaErrors(
        cudaStreamCreateWithPriority( &cuda_streams_[0],
                                      cudaStreamNonBlocking,
                                      priority_high ) );  // Main stream
    checkCudaErrors(
        cudaStreamCreateWithPriority( &cuda_streams_[1],
                                      cudaStreamNonBlocking,
                                      priority_high ) );  // resampling Up

    // Create events
    for ( int i = 0; i < kNumEvents; i++ ) {
        checkCudaErrors( cudaEventCreateWithFlags( &cuda_events_[i],
                                                   cudaEventDisableTiming ) );
    }

    // Copy Models to pinned memory
    std::memcpy( h_pin_sq_initial_noise_cov_.get( ),
                 const_cast<T *>( models::kSqInitialNoiseCov.val.data( ) ),
                 kSysDim * kSysDim * sizeof( T ) );

    std::memcpy( h_pin_inv_meas_noise_cov_.get( ),
                 const_cast<T *>( models::kInvMeasNoiseCov.val.data( ) ),
                 kMeasDim * kMeasDim * sizeof( T ) );

    std::memcpy( h_pin_sq_process_noise_cov_.get( ),
                 const_cast<T *>( models::kSqProcessNoiseCov.val.data( ) ),
                 kSysDim * kSysDim * sizeof( T ) );
}

template<typename T>
ParticleBpfGpu<T>::~ParticleBpfGpu( ) noexcept {

    for ( int i = 0; i < kNumStreams; i++ ) {
        checkCudaErrors( cudaStreamDestroy( cuda_streams_[i] ) );
    }

    for ( int i = 0; i < kNumEvents; i++ ) {
        checkCudaErrors( cudaEventDestroy( cuda_events_[i] ) );
    }
}

template<typename T>
void ParticleBpfGpu<T>::Initialize(
    int const &                      mcs,
    std::vector<std::vector<float>> &timing_results ) {

    RANGE( "ParticleBpfGpu", color++ )

    // Compute initial state for all particles
    // Create randomly generated particles from the initial prior Gaussian
    // distribution
    InitializeParticles( );

    // For timing purposes
    std::vector<T> time_vector( kSamples, 0.0f );
    cudaEvent_t    start {}, stop {};
    checkCudaErrors( cudaEventCreate( &start ) );
    checkCudaErrors( cudaEventCreate( &stop ) );

    for ( int i = 0; i < kSamples; i++ ) {
        checkCudaErrors( cudaEventRecord( start ) );

        UpdateTimeStep( );

        checkCudaErrors( cudaEventRecord( stop ) );
        checkCudaErrors( cudaEventSynchronize( stop ) );
        float milliseconds { 0.0f };
        checkCudaErrors( cudaEventElapsedTime( &milliseconds, start, stop ) );
        time_vector[i] = milliseconds * 1000;
        time_step++;
    }

    utility::ComputeOverallTiming(
        time_vector, median, mean, stdDev );  // Compute standard deviation

    timing_results[static_cast<int>( utility::Timing::kMedian )][mcs] = median;
    timing_results[static_cast<int>( utility::Timing::kMean )][mcs]   = mean;
    timing_results[static_cast<int>( utility::Timing::kStdDev )][mcs] = stdDev;
}

template<typename T>
void ParticleBpfGpu<T>::InitializeParticles( ) {

    RANGE( "InitializeParticles", color++ )

    InitializeFilterCuda<T>( sm_count_,
                             cuda_streams_,
                             kNumParticles,
                             h_pin_sq_initial_noise_cov_.get( ),
                             cuda_events_,
                             d_particle_state_new_.get( ) );

    checkCudaErrors( cudaDeviceSynchronize( ) );
}

template<typename T>
void ParticleBpfGpu<T>::UpdateTimeStep( ) {

    RANGE( "UpdateTimeStep", color++ )

    RetrieveMeasurement( );
    ComputeMeasErrors( );
    ComputeEstimates( );
    ComputeResampleIndex( );
    ComputeParticleTransition( );
    checkCudaErrors( cudaDeviceSynchronize( ) );
}

template<typename T>
void ParticleBpfGpu<T>::RetrieveMeasurement( ) {

    color = 5;  // Reset color each Monte Carlo
    RANGE( "RetrieveMeasurement", color++ )

    // Read in measurement data each time step
    int idx { time_step * kMeasDim };
    truth_meas_func_ptr_( idx, h_pin_meas_update_.get( ) );
}

template<typename T>
void ParticleBpfGpu<T>::ComputeMeasErrors( ) {

    RANGE( "ComputeMeasErrors", color++ )

    /*
     * Current measurement minus computed measurement function based on current
     * particle state for all particle sets Copy d_particle_state_new_ to
     * d_particle_state_
     */
    ComputeMeasErrorsCuda<T>( sm_count_,
                              cuda_streams_,
                              kNumParticles,
                              h_pin_inv_meas_noise_cov_.get( ),
                              h_pin_meas_update_.get( ),
                              d_particle_state_new_.get( ),
                              cuda_events_,
                              d_particle_weights_.get( ),
                              d_particle_state_.get( ) );
}

template<typename T>
void ParticleBpfGpu<T>::ComputeEstimates( ) {

    RANGE( "ComputeEstimates", color++ )

    ComputeEstimatesCuda<T>( sm_count_,
                             cuda_streams_,
                             kNumParticles,
                             time_step,
                             kResamplingMethod,
                             d_particle_state_.get( ),
                             cuda_events_,
                             d_filtered_estimates_.get( ),
                             d_particle_weights_.get( ) );
}

template<typename T>
void ParticleBpfGpu<T>::ComputeResampleIndex( ) {

    RANGE( "ComputeResampleIndex", color++ )

    ComputeResampleIndexCuda<T>( sm_count_,
                                 cuda_streams_,
                                 kNumParticles,
                                 time_step,
                                 kResamplingMethod,
                                 d_particle_weights_.get( ),
                                 d_prefix_sum_particle_weights_.get( ),
                                 cuda_events_,
                                 d_resampling_index_up_.get( ),
                                 d_resampling_index_down_.get( ) );
}

template<typename T>
void ParticleBpfGpu<T>::ComputeParticleTransition( ) {

    RANGE( "ComputeParticleTransition", color++ )

    ComputeParticleTransitionCuda<T>( sm_count_,
                                      cuda_streams_,
                                      kNumParticles,
                                      kResamplingMethod,
                                      h_pin_sq_process_noise_cov_.get( ),
                                      d_particle_state_.get( ),
                                      d_resampling_index_up_.get( ),
                                      d_resampling_index_down_.get( ),
                                      cuda_events_,
                                      d_particle_state_new_.get( ) );
}

template<typename T>
void ParticleBpfGpu<T>::WriteOutput( const std::string &filename ) {

    // Copy device memory to host
    checkCudaErrors( cudaMemcpyAsync( filtered_estimates_.data( ),
                                      d_filtered_estimates_.get( ),
                                      kSysDim * kSamples * sizeof( T ),
                                      cudaMemcpyDeviceToHost,
                                      cuda_streams_[0] ) );

    // Create iterator for filtered estimates
    typename std::vector<T>::const_iterator it = filtered_estimates_.cbegin( );
    utility::WriteToFile( filename, "bpf", kSamples, median, mean, stdDev, it );
}

template<typename T>
template<typename U>
U *ParticleBpfGpu<T>::DeviceAllocate( int const &N ) {
    U *    ptr { nullptr };
    size_t bytes { N * sizeof( U ) };
    checkCudaErrors( cudaMalloc( reinterpret_cast<void **>( &ptr ), bytes ) );
    checkCudaErrors( cudaMemset( ptr, 0, bytes ) );
    return ( ptr );
}

template<typename T>
template<typename U>
void ParticleBpfGpu<T>::DeviceMemoryDeleter<U>::operator( )( U *ptr ) {
    if ( ptr ) {
        checkCudaErrors( cudaFree( ptr ) );
    }
};

template<typename T>
template<typename U>
U *ParticleBpfGpu<T>::HostAllocate( int const &N ) {
    U *    ptr { nullptr };
    size_t bytes { N * sizeof( U ) };
    checkCudaErrors( cudaHostAlloc( reinterpret_cast<void **>( &ptr ),
                                    bytes,
                                    cudaHostAllocWriteCombined ) );
    return ( ptr );
}

template<typename T>
template<typename U>
void ParticleBpfGpu<T>::HostMemoryDeleter<U>::operator( )( U *ptr ) {
    if ( ptr ) {
        checkCudaErrors( cudaFreeHost( ptr ) );
    }
};

} /* namespace filters */
