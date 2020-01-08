/**
 * @file particle_bpf_gpu.h
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains header information for ParticleBpfGpu class.
 *
 * @copyright Copyright (c) 2020
 *
 * @license This project is released under the GNU Public License
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */

#ifndef Particle_BPF_GPU_H_
#define Particle_BPF_GPU_H_

#include <functional>  // std::function
#include <string>      // std::string
#include <vector>      // std::vector

#include "particle_base.h"
#include "utilities.h"

namespace filters {

constexpr int kNumStreams { 2 };
constexpr int kNumEvents { 2 };

/**
 * @class ParticleBpfGpu
 * @brief Bootstrap Particle Filter GPU class
 * @tparam T Input should be of type float
 */
template<typename T>
class ParticleBpfGpu : public ParticleBase {
  public:
    /**
     * @brief Construct a new ParticleBpfGpu object
     *
     */
    ParticleBpfGpu( ) noexcept = delete;

    /**
     * @brief Construct a new ParticleBpfGpu object
     *
     * @param[in] info Structure containing information about filter
     * @param[in] truth_meas_func_ptr Pointer to true measurement data
     */
    ParticleBpfGpu(
        utility::FilterInfo const &filter_info,
        std::function<void( int const &idx, float *current_meas_data )> const
            &truth_meas_func_ptr );

    /**
     * @brief Destroy the ParticleBpfGpu object
     *
     */
    virtual ~ParticleBpfGpu( ) noexcept;

    /**
     * @brief Copy construct a new ParticleBpfGpu object
     *
     */
    explicit ParticleBpfGpu( ParticleBpfGpu const & ) = delete;

    /**
     * @brief Copy assignment construct a new ParticleBpfGpu object
     *
     */
    ParticleBpfGpu &operator=( ParticleBpfGpu const & ) = delete;

    /**
     * @brief Move construct a new ParticleBpfGpu object
     *
     */
    explicit ParticleBpfGpu( ParticleBpfGpu && ) noexcept = delete;

    /**
     * @brief Move assignment construct a new ParticleBpfGpu object
     *
     */
    ParticleBpfGpu &operator=( ParticleBpfGpu && ) noexcept = delete;

    /**
     * @brief Initialize filter and begin processing
     *
     * @param[in] mcs Monte Carlo being executed
     * @param[out] timing_results 2D vector to hold timing results of all Monte
     * Carlo runs
     */
    void Initialize( int const &                      mcs,
                     std::vector<std::vector<float>> &timing_results );

    /**
     * @brief Function to write filter estimates to output file
     *
     * @param[in] filename Output filename
     */
    void WriteOutput( const std::string &filename );

  private:
    /**
     * @brief Initial particles based on initial noise covariance
     *
     */
    void InitializeParticles( );

    /**
     * @brief UpdateTimeStep filter at each time step
     *
     */
    void UpdateTimeStep( );

    /**
     * @brief Receive actual measurements from sensors
     *
     */
    void RetrieveMeasurement( );

    /**
     * @brief Compute the measurement error
     */
    void ComputeMeasErrors( );

    /**
     * @brief Compute filtered estimates
     *
     */
    void ComputeEstimates( );

    /**
     * @brief Compute resampling index
     *
     */
    void ComputeResampleIndex( );

    /**
     * @brief Predict the particle one step forward
     *
     */
    void ComputeParticleTransition( );

    /**
     * @brief Memory allocation on device. Followed by memory
     * set to zero.
     *
     * @tparam U Data type of allocation
     * @param[in] N Number of elements in array
     * @return U* A pointer to allocation
     */
    template<typename U>
    U *DeviceAllocate( int const &N );

    /**
     * @brief Deleter for device memory allocations
     *
     * @tparam U Data type of allocation
     */
    template<typename U>
    struct DeviceMemoryDeleter {
        void operator( )( U *ptr );
    };

    /**
     * @brief Smart pointer for RAII device allocation
     * and destruction
     *
     * @tparam U
     */
    template<typename U>
    using UniqueDevicePtr = std::unique_ptr<U, DeviceMemoryDeleter<U>>;

    /**
     * @brief Pinned memory allocation on host.
     *
     * @tparam U Data type of allocation
     * @param[in] N Number of elements in array
     * @return U* A pointer to allocation
     */
    template<typename U>
    U *HostAllocate( int const &N );

    /**
     * @brief Deleter for pinned host memory allocations
     *
     * @tparam U Data type of allocation
     */
    template<typename U>
    struct HostMemoryDeleter {
        void operator( )( U *ptr );
    };

    /**
     * @brief Smart pointer for RAII device allocation
     * and destruction
     *
     * @tparam U
     */
    template<typename U>
    using UniqueHostPtr = std::unique_ptr<U, HostMemoryDeleter<U>>;

    int const kSysDim;       /**< Quantity of dimensions of the system */
    int const kMeasDim;      /**< Quantity of dimensions of the measurement */
    int const kSamples;      /**< Quantity of samples to be processed */
    int const kNumParticles; /**< The quantity of particles */
    int const kResamplingMethod; /**< Type of filter: Bootstrap = 0 */

    std::vector<T> filtered_estimates_; /**< Vector to store sample estimations
                                           : sysDim * samples */

    std::function<void( int const &idx, float *current_meas_data )>
        truth_meas_func_ptr_; /**< Function pointer to true measurement data */

    int device_id_; /**< GPU device id */
    int sm_count_;  /**< Number of multiprocessors on device */

    /*
     * Device variables
     */
    cudaStream_t cuda_streams_[kNumStreams]; /**< Pointer to CUDA stream */
    cudaEvent_t  cuda_events_[kNumEvents];   /**< Event handle for CUDA */

    UniqueDevicePtr<T> d_filtered_estimates_; /**< Device storage : filtered
                                                 estimates : samples * sysDim */
    UniqueDevicePtr<T> d_particle_state_; /**< Device storage : current particle
                                             state : particles * sysDim */
    UniqueDevicePtr<T>
        d_particle_state_new_; /**< Device storage : current particle state :
                                  particles * sysDim */
    UniqueDevicePtr<T> d_particle_weights_; /**< Device storage : particle
                                               weights : particles */
    UniqueDevicePtr<T>
        d_prefix_sum_particle_weights_; /**< Device storage : prefix-sum weights
                                           : particles */
    UniqueDevicePtr<int>
        d_resampling_index_up_; /**< Device storage : resampling index :
                                   particles */
    UniqueDevicePtr<int>
        d_resampling_index_down_; /**< Device storage : resampling index :
                                     particles */

    /*
     * Pinned Memory
     */
    UniqueHostPtr<T>
        h_pin_sq_initial_noise_cov_; /**< Squared covariance of the initial
                                        state : sysDim * sysDim */
    UniqueHostPtr<T>
        h_pin_inv_meas_noise_cov_; /**< Inverse measurement noise covariance :
                                      measDim * measDim */
    UniqueHostPtr<T>
        h_pin_sq_process_noise_cov_;     /**< Squared process noise covariance :
                                            sysDim * sysDim */
    UniqueHostPtr<T> h_pin_meas_update_; /**< Store measurement UpdateTimeStep
                                            at time step : measDim */
};

} /* namespace filters */

#include "particle_bpf_gpu_impl.h"

#endif /* Particle_BPF_GPU_H_ */
