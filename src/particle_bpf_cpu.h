/**
 * @file particle_bpf_cpu.h
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

#ifndef Particle_BPF_CPU_H_
#define Particle_BPF_CPU_H_

#include <functional>  // std::function
#include <string>      // std::string
#include <vector>      // std::vector

#include "particle_base.h"
#include "utilities.h"

/**
 * @brief Contains functions for filter functionality
 */
namespace filters {

/**
 * @class ParticleBpfCpu
 * @brief Serial version of the bootstrap particle filter
 *
 * @see http://www.irisa.fr/aspi/legland/ref/arulampalam02a.pdf
 */

class ParticleBpfCpu : public ParticleBase {
  public:
    /**
     * @brief Construct a new ParticleBpfCpu object
     *
     */
    ParticleBpfCpu( ) noexcept = default;

    /**
     * @brief Construct a new ParticleBpfCpu object
     *
     * @param[in] info Structure containing information about filter
     * @param[in] truth_meas_func_ptr Pointer to true measurement data
     */

    ParticleBpfCpu(
        utility::FilterInfo const &filter_info,
        std::function<void( int const &idx, float *current_meas_data )> const
            &truth_meas_func_ptr );

    /**
     * @brief Destroy the ParticleBpfCpu object
     *
     */
    virtual ~ParticleBpfCpu( ) noexcept;

    /**
     * @brief Copy construct a new ParticleBpfCpu object
     *
     */
    explicit ParticleBpfCpu( ParticleBpfCpu const & ) = delete;

    /**
     * @brief Copy assignment construct a new ParticleBpfCpu object
     *
     */
    ParticleBpfCpu &operator=( ParticleBpfCpu const & ) = delete;

    /**
     * @brief Move construct a new ParticleBpfCpu object
     *
     */
    explicit ParticleBpfCpu( ParticleBpfCpu && ) noexcept = delete;

    /**
     * @brief Move assignment construct a new ParticleBpfCpu object
     *
     */
    ParticleBpfCpu &operator=( ParticleBpfCpu && ) noexcept = delete;

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
    void WriteOutput( std::string const &filename );

  private:
    /**
     * @brief UpdateTimeStep filter at each time step
     *
     */
    void UpdateTimeStep( );

    /**
     * @brief Initial particles based on initial noise covariance
     *
     */
    void InitializeParticles( );

    /**
     * @brief Receive actual measurements from sensors
     *
     */
    void RetrieveMeasurement( );

    /**
     * @brief Compute the measurement error
     *
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

    int const kSysDim;       /**< Quantity of dimensions of the system */
    int const kMeasDim;      /**< Quantity of dimensions of the measurement */
    int const kSamples;      /**< Quantity of samples to be processed */
    int const kNumParticles; /**< The quantity of particles */
    int const kResamplingMethod; /**< Type of filter: Bootstrap = 0 */

    std::vector<float> filtered_estimates_; /**< Vector to store sample
                                               estimations : sysDim * samples */
    std::vector<float> meas_update_;        /**< Vector to store measurement
                                               Updatetime_step at time step */
    std::vector<float>
                       meas_estimates_; /**< Vector to store measurement updates : measDim */
    std::vector<float> meas_errors_; /**< Vector to store measurement errors :
                                        measDim * particle */
    std::vector<float> particle_state_;  /**< Vector to store particle state :
                                            sysDim * particles */
    std::vector<float> particle_weight_; /**< Vector to store particle weights :
                                            1 * particles */
    std::vector<int> resampling_index_;  /**< Vector to store with particles to
                                            be resampled : 1 * particles */

    std::function<void( int const &idx, float *current_meas_data )>
        truth_meas_func_ptr_; /**< Function pointer to true measurement data */
};

} /* namespace filters */

#endif /* Particle_BPF_CPU_H_ */
