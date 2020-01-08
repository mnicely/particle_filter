/**
 * @file particle_base.h
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains the parent ParticleBase class
 * for particle filter
 *
 * @copyright Copyright (c) 2020
 *
 * @license This project is released under the GNU Public License
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */

#ifndef ParticleBase_H_
#define ParticleBase_H_

/**
 * @brief Contains functions for filter functionality
 */
namespace filters {

/**
 * @class ParticleBase
 * @brief Particle Filter base class
 */
class ParticleBase {
  public:
    /**
     * @brief Construct a new ParticleBase object
     *
     */
    ParticleBase( ) noexcept {}

    /**
     * @brief Destroy the ParticleBase object
     *
     */
    virtual ~ParticleBase( ) noexcept {}

    /**
     * @brief Copy construct a new ParticleBase object
     *
     */
    explicit ParticleBase( ParticleBase const & ) = delete;

    /**
     * @brief Copy assignment construct a new ParticleBase object
     *
     */
    ParticleBase &operator=( ParticleBase const & ) = delete;

    /**
     * @brief Move construct a new ParticleBase object
     *
     */
    explicit ParticleBase( ParticleBase && ) noexcept = delete;

    /**
     * @brief Move assignment construct a new ParticleBase object
     *
     */
    ParticleBase &operator=( ParticleBase && ) noexcept = delete;

    /**
     * @brief Initialize filter and begin processing
     *
     * @param[in] num_mcs Monte Carlo being executed
     * @param[out] timing_results 2D vector to hold timing results of all Monte Carlo runs
     */
    virtual void Initialize( int const &num_mcs, std::vector<std::vector<float>> &timing_results ) = 0;

    /**
     * @brief UpdateTimeStep filter at a given time step
     *
     */
    virtual void UpdateTimeStep( ) = 0;

    /**
     * @brief Initial particles based on initial noise covariance
     *
     */
    virtual void InitializeParticles( ) = 0;

    /**
     * @brief Receive actual measurements from sensors
     *
     */
    virtual void RetrieveMeasurement( ) = 0;

    /**
     * @brief Compute measurement error
     *
     */
    virtual void ComputeMeasErrors( ) = 0;

    /**
     * @brief Compute filtered estimates
     *
     */
    virtual void ComputeEstimates( ) = 0;

    /**
     * @brief Compute resampling index
     *
     */
    virtual void ComputeResampleIndex( ) = 0;

    /**
     * @brief Predict the particle one step forward
     *
     */
    virtual void ComputeParticleTransition( ) = 0;

    /**
     * @brief Write filtered estimates to file
     *
     * @param[in] filename Filename of file where filtered estimates are stored
     */
    virtual void WriteOutput( const std::string &filename ) = 0;

    int time_step {}; /**< Current time step */
    int color {};     /**< Holds color value for NVTX profiling */

    float median {}; /**< Timing median across all Monte Carlos */
    float mean {};   /**< Timing mean across all Monte Carlos */
    float stdDev {}; /**< Timing standard deviation across all Monte Carlos */
};

} /* namespace filters */

#endif /* ParticleBase_H_ */
