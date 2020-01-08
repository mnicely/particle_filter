/**
 * @file particle_bpf_cpu.cpp
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains implementation of serial version of the
 * bootstrap particle filter.
 *
 * @copyright Copyright (c) 2020
 *
 * @license This project is released under the GNU Public License
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */

#include <algorithm>  // std::transform, std::remove_if, std::nth_element, std::fill, std::for_each
#include <chrono>      // std::chrono
#include <cmath>       // std::sqrt, std::exp
#include <functional>  // std::multiplies
#include <iterator>    // std::vector<>::const_iterator
#include <numeric>  // std::accumulate, std::inner_product, std::partial_sum, std::iota
#include <random>  // std::default_random_engine, std::random_device, std::uniform_real_distribution

#include "models.h"
#include "particle_bpf_cpu.h"

namespace filters {

ParticleBpfCpu::ParticleBpfCpu(
    utility::FilterInfo const &filter_info,
    std::function<void( int const &idx, float *current_meas_data )> const
        &truth_meas_func_ptr ) :
    // clang-format off
    kSysDim { utility::kSysDim }, 
    kMeasDim { utility::kMeasDim },
    kSamples { filter_info.samples },
    kNumParticles { filter_info.particles },
    kResamplingMethod { filter_info.resampling },
    filtered_estimates_( kSysDim * filter_info.samples, 0.0f ), 
    meas_update_( kMeasDim, 0.0f ),
    meas_estimates_( kMeasDim, 0.0f ), 
    meas_errors_( kMeasDim * kNumParticles, 0.0f ),
    particle_state_( kSysDim * kNumParticles, 0.0f ), 
    particle_weight_( kNumParticles, 0.0f ),
    resampling_index_( kNumParticles, 0.0f ), 
    truth_meas_func_ptr_ { truth_meas_func_ptr }  // clang-format on
{}

ParticleBpfCpu::~ParticleBpfCpu( ) noexcept {}

void ParticleBpfCpu::Initialize(
    int const &                      mcs,
    std::vector<std::vector<float>> &timing_results ) {

    RANGE( "ParticleBpfCpu", color++ )

    // Compute initial state for all particles
    // Create randomly generated particles from the initial prior Gaussian
    // distribution
    InitializeParticles( );

    std::vector<float> time_vector( kSamples, 0 );

    for ( int i = 0; i < kSamples; i++ ) {
        auto start = std::chrono::high_resolution_clock::now( );
        UpdateTimeStep( );
        auto end = std::chrono::high_resolution_clock::now( );
        auto nanos =
            std::chrono::duration_cast<std::chrono::microseconds>( end - start )
                .count( );
        time_vector[i] = static_cast<float>( nanos );
        time_step++;  // Increment step
    }

    utility::ComputeOverallTiming( time_vector, median, mean, stdDev );

    timing_results[static_cast<int>( utility::Timing::kMedian )][mcs] = median;
    timing_results[static_cast<int>( utility::Timing::kMean )][mcs]   = mean;
    timing_results[static_cast<int>( utility::Timing::kStdDev )][mcs] = stdDev;
}

void ParticleBpfCpu::InitializeParticles( ) {

    RANGE( "InitializeParticles", color++ )

    utility::Matrix initial_cov { models::kInitialNoiseCov };
    utility::Matrix initial_state { models::kInitialState };

    // Compute square root of covariance matrix of initial state
    utility::MatrixSqrt( initial_cov );

    // Create vector to store random numbers
    std::vector<float> random_numbers( kNumParticles * kSysDim, 0.0f );

    // Create a submatrix to hold random numbers : system dimensions * 1
    utility::Matrix subset_random_numbers { kSysDim, 1 };

    // Create matrix to store squared initial noise covariance matrix times
    // random numbers
    utility::Matrix matrix_mult_result { kSysDim, kSysDim };

    // Generate random numbers
    utility::GenerateRandomNum( random_numbers,
                                models::kSysDistrLowerLimit,
                                models::kSysDistrUpperLimit );

    // For all particle sets
    for ( int i = 0; i < kNumParticles; i++ ) {

        // Replace submatrix values with new random numbers
        for ( int j = 0; j < kSysDim; j++ ) {
            subset_random_numbers.val[j] = random_numbers[kSysDim * i + j];
        }

        // Perform matrix multiplication squared noise covariance matrix times
        utility::MatrixMult(
            matrix_mult_result,
            initial_cov,
            subset_random_numbers );  // <--- Error here with OpenMP

        // Store noise covariance matrix * randon numbers
        for ( int j = 0; j < kSysDim; j++ ) {
            particle_state_[i * kSysDim + j] =
                initial_state.val[j] + matrix_mult_result.val[j];
        }
    }
}

void ParticleBpfCpu::UpdateTimeStep( ) {

    RANGE( "UpdateTimeStep", color++ )

    RetrieveMeasurement( );
    ComputeMeasErrors( );
    ComputeEstimates( );
    ComputeResampleIndex( );
    ComputeParticleTransition( );
}

void ParticleBpfCpu::RetrieveMeasurement( ) {

    RANGE( "RetrieveMeasurement", color++ )

    // Create temporary vector to grab data
    int idx { time_step * kMeasDim };
    truth_meas_func_ptr_( idx, meas_update_.data( ) );
}

void ParticleBpfCpu::ComputeMeasErrors( ) {

    RANGE( "ComputeMeasErrors", color++ )

    /*
     * Current measurement minus computed measurement function based on current
     * particle state For all particle sets
     */
    for ( int i = 0; i < kNumParticles; i++ ) {

        // Index through particle set
        int idx {
            i * kSysDim
        };  // Use sysDim because we are indexing through particle data

        // Perform measurement functions
        models::MeasModelMath( &particle_state_[idx], meas_estimates_.data( ) );

        for ( int j = 0; j < kMeasDim; j++ ) {
            meas_errors_[i * kMeasDim + j] =
                meas_update_[j] - meas_estimates_[j];
        }
    }
}

void ParticleBpfCpu::ComputeEstimates( ) {

    RANGE( "ComputeEstimates", color++ )

    // Create matrix to store inverse of meas_noise_cov
    utility::Matrix inv_meas_noise_cov { models::kMeasNoiseCov };

    // Calculate inverse of noise covariance
    utility::ComputeInverse( inv_meas_noise_cov );

    // Square Measurement Error
    std::transform( meas_errors_.begin( ),
                    meas_errors_.end( ),
                    meas_errors_.begin( ),
                    meas_errors_.begin( ),
                    std::multiplies<float>( ) );

    // Store measurementErrors^2 in matrix for following math
    utility::Matrix meas_errors_sq { kMeasDim, kNumParticles, meas_errors_ };

    // Create matrix to store measurement error squared divided by the inverse
    utility::Matrix meas_errors_sq_div_inv {
        kMeasDim, static_cast<int>( meas_errors_.size( ) )
    };

    // Matrix multiplication inverse of noise covariance times squared
    // measurement error
    utility::MatrixMult(
        meas_errors_sq_div_inv, inv_meas_noise_cov, meas_errors_sq );

    // Clear old values
    std::fill( particle_weight_.begin( ), particle_weight_.end( ), 0 );

    // Sum particle sets
    for ( int i = 0; i < kNumParticles; i++ ) {
        for ( int j = 0; j < kMeasDim; j++ ) {
            particle_weight_[i] += meas_errors_sq_div_inv.val[i * kMeasDim + j];
        }
        particle_weight_[i] = std::exp( particle_weight_[i] * -0.5f );
    }

    // Normalize weights
    float sum_of_particle_weights = std::accumulate(
        particle_weight_.begin( ), particle_weight_.end( ), 0.0f );

    //	 Check for filter divergence
    if ( sum_of_particle_weights < 1e-12f ) {
        std::printf( "BPF is diverging @ %d\n", time_step );
    }

    // Normalize the importance weights
    std::transform( particle_weight_.begin( ),
                    particle_weight_.end( ),
                    particle_weight_.begin( ),
                    [&sum_of_particle_weights]( float &a ) {
                        return ( a / sum_of_particle_weights );
                    } );

    for ( int i = 0; i < kSysDim; i++ ) {
        for ( int j = 0; j < kNumParticles; j++ ) {
            filtered_estimates_[i + time_step * kSysDim] +=
                particle_state_[j * kSysDim + i] * particle_weight_[j];
        }
    }
}

void ParticleBpfCpu::ComputeResampleIndex( ) {

    RANGE( "ComputeResampleIndex", color++ )

    // Create vector to store cumulative normal weights
    std::vector<float> prefix_sum_particle_weights( kNumParticles, 0.0f );

    // Compute cumulative summation on normalized weights
    std::partial_sum( particle_weight_.begin( ),
                      particle_weight_.end( ),
                      prefix_sum_particle_weights.begin( ),
                      std::plus<float>( ) );

    // Because of rounding error make last element in
    // prefix_sum_particle_weights = 1.0 This is to resolve infinite while loop
    // below
    prefix_sum_particle_weights.back( ) = 1.0f;

    // Create vector that increments for 0 to N (# of particles)
    std::vector<float> uniform_random_numbers( kNumParticles, 0.0f );
    std::iota(
        uniform_random_numbers.begin( ), uniform_random_numbers.end( ), 0.0f );

    std::default_random_engine       gen( std::random_device {}( ) );
    std::uniform_real_distribution<> distr( 0, 1 );

    if ( kResamplingMethod ==
         static_cast<int>( utility::Method::kSystematic ) ) {
        std::transform( uniform_random_numbers.begin( ),
                        uniform_random_numbers.end( ),
                        uniform_random_numbers.begin( ),
                        std::bind1st( std::plus<float>( ), distr( gen ) ) );
    } else if ( kResamplingMethod ==
                static_cast<int>( utility::Method::kStratified ) ) {
        std::for_each(
            uniform_random_numbers.begin( ),
            uniform_random_numbers.end( ),
            [&distr, &gen]( float &a ) { return ( a += distr( gen ) ); } );
    }

    // Divide each element by N (# of particles)
    std::transform( uniform_random_numbers.begin( ),
                    uniform_random_numbers.end( ),
                    uniform_random_numbers.begin( ),
                    [&]( float &a ) { return ( a / kNumParticles ); } );

    int k {};
    for ( int i = 0; i < kNumParticles; i++ ) {
        while ( prefix_sum_particle_weights[k] < uniform_random_numbers[i] ) {
            k++;
        }
        resampling_index_[i] = k;
    }

    std::vector<float> temp_storage( particle_state_.size( ), 0.0f );
    for ( int i = 0; i < kNumParticles; i++ ) {
        for ( int j = 0; j < kSysDim; j++ ) {
            temp_storage[i * kSysDim + j] =
                particle_state_[resampling_index_[i] * kSysDim + j];
        }
    }

    particle_state_ = temp_storage;
}

void ParticleBpfCpu::ComputeParticleTransition( ) {

    RANGE( "ComputeParticleTransition", color++ )

    utility::Matrix process_noise_cov { models::kProcessNoiseCov };

    // Compute square root of noise covariance matrix
    utility::MatrixSqrt( process_noise_cov );

    // Create vector to hold random numbers and process function
    std::vector<float> random_numbers( kNumParticles * kSysDim, 0.0f );
    std::vector<float> temp_vector( kSysDim, 0.0f );

    // Create a submatrix for each sample set
    utility::Matrix subset_random_numbers { kSysDim, 1 };

    // Create matrix to store squared measurement noise covariance matrix times
    // random numbers
    utility::Matrix matrix_mult_result { kSysDim, kSysDim };

    // Generate random numbers
    utility::GenerateRandomNum( random_numbers,
                                models::kSysDistrLowerLimit,
                                models::kSysDistrUpperLimit );

    // For all sample sets
    for ( int i = 0; i < kNumParticles; i++ ) {

        // Replace submatrix values with new random numbers
        for ( int j = 0; j < kSysDim; j++ ) {
            subset_random_numbers.val[j] = random_numbers[kSysDim * i + j];
        }

        // Perform matrix multiplication squared process noise covariance matrix
        // times random numbers
        utility::MatrixMult(
            matrix_mult_result, process_noise_cov, subset_random_numbers );

        // Compute system function at each time step based on previous time step
        // Store result in temp_vector
        models::SysModelMath( &particle_state_[i * kSysDim],
                              temp_vector.data( ) );

        for ( int j = 0; j < kSysDim; j++ ) {
            particle_state_[i * kSysDim + j] =
                temp_vector[j] + matrix_mult_result.val[j];
        }
    }
}

void ParticleBpfCpu::WriteOutput( std::string const &filename ) {

    // Create iterator for filtered estimates
    typename std::vector<float>::const_iterator it =
        filtered_estimates_.cbegin( );
    utility::WriteToFile( filename, "bpf", kSamples, median, mean, stdDev, it );
}

} /* namespace filters */
