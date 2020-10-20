/**
 * @file generate_data.cpp
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains implementation to generate truth data.
 *
 * @copyright Copyright (c) 2020
 *
 * @license This project is released under the GNU Public License
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */

#include <fstream>    // std::ifstream
#include <ios>        // std::ios_base
#include <iostream>   // std::ignore
#include <iterator>   // std::vector<>::const_iterator
#include <sstream>    // std::istringstream
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string, std::getline, std::to_string
#include <vector>     // std::vector

#include "generate_data.h"

#include "models.h"

/**
 * @brief Contains functions for filter functionality
 *
 */
namespace filters {

GenerateData::GenerateData( utility::FilterInfo const &filter_info ) :
    // clang-format off
    kSysDim { utility::kSysDim }, 
    kMeasDim { utility::kMeasDim }, 
    kSamples { filter_info.samples },
    kNumMcs { filter_info.num_mcs },
    system_truth_data_ ( kSysDim * filter_info.samples, 0.0f ),
    meas_truth_data_ ( kMeasDim * filter_info.samples, 0.0f )
// clang-format on
{}

GenerateData::~GenerateData( ) noexcept {}

void GenerateData::get_current_meas_data( int const &idx, float *current_meas_data ) {
    for ( int i = 0; i < kMeasDim; i++ ) {
        current_meas_data[i] = meas_truth_data_[idx + i];
    }
}

// Need to create data directory
void GenerateData::CreateTruthData( std::string const &system_data_file, std::string const &meas_data_file ) {

    if ( !std::remove( system_data_file.c_str( ) ) ) {
        std::printf( "Previous %s successfully removed\n", system_data_file.c_str( ) );
    }
    if ( !std::remove( meas_data_file.c_str( ) ) ) {
        std::printf( "Previous %s successfully removed\n", meas_data_file.c_str( ) );
    }

    utility::WriteTruthHeader( system_data_file, kNumMcs, kSysDim, kSamples );
    utility::WriteTruthHeader( meas_data_file, kNumMcs, kMeasDim, kSamples );

    for ( int i = 0; i < kNumMcs; i++ ) {
        GenerateTruthData( );
        WriteCreatedTruthData( system_data_file, meas_data_file );
    }

    std::printf(
        "Successfullly created truth data in %s and %s.\n", system_data_file.c_str( ), meas_data_file.c_str( ) );
    std::printf( "With %d Monte Carlo runs and %d samples per run.\n", kNumMcs, kSamples );

    exit( EXIT_SUCCESS );
}

void GenerateData::GenerateTruthData( ) {

    // Generate truth system data
    GenerateSystemTruthData( );

    // Generate truth measurement data
    GenerateMeasTruthData( );
}

void GenerateData::GenerateSystemTruthData( ) {

    utility::Matrix initial_cov { models::kInitialNoiseCov };
    utility::Matrix process_noise_cov { models::kProcessNoiseCov };
    utility::Matrix initial_state { models::kInitialState };

    // Compute square root of covariance matrix of initial state
    utility::MatrixSqrt( initial_cov );

    // Compute square root of noise covariance matrix
    utility::MatrixSqrt( process_noise_cov );

    // Create vector to hold random numbers
    std::vector<float> random_numbers( kSamples * kSysDim, 0.0f );

    // Create a submatrix to hold random numbers : system dimensions * 1
    utility::Matrix subset_random_numbers { kSysDim, 1 };

    // Create matrix to store squared initial noise covariance matrix times
    // random numbers
    utility::Matrix matrix_mult_result { kSysDim, 1 };

    // Generate random numbers
    utility::GenerateRandomNum( random_numbers, models::kSysDistrLowerLimit, models::kSysDistrUpperLimit );

    // Copy first set of random numbers to matrix data
    for ( int j = 0; j < kSysDim; j++ ) {
        subset_random_numbers.val[j] = random_numbers[0 * kSysDim + j];
    }

    // Perform matrix multiplication squared process noise covariance matrix
    // times random numbers
    utility::MatrixMult( matrix_mult_result, process_noise_cov, subset_random_numbers );

    // Store initial state + (initial noise covariance matrix * randon numbers)
    // in truthSystemData
    for ( int j = 0; j < kSysDim; j++ ) {
        system_truth_data_[0 * kSysDim + j] = initial_state.val[j] + matrix_mult_result.val[j];
    }

    // For all sample sets
    for ( int i = 1; i < kSamples; i++ ) {

        // Copy next set of random numbers to matrix data
        for ( int j = 0; j < kSysDim; j++ ) {
            subset_random_numbers.val[j] = random_numbers[i * kSysDim + j];
        }

        // Perform matrix multiplication squared process noise covariance matrix
        // times random numbers
        utility::MatrixMult( matrix_mult_result, process_noise_cov, subset_random_numbers );

        // Once we've completed matrix multiplcation we need to process system
        // function with information from the previous state
        models::SysModelMath( &system_truth_data_[( i - 1 ) * kSysDim], &system_truth_data_[i * kSysDim] );

        // Store system function + (noise covariance matrix * randon numbers)
        for ( int j = 0; j < kSysDim; j++ ) {
            system_truth_data_[i * kSysDim + j] += matrix_mult_result.val[j];
        }
    }
}

void GenerateData::GenerateMeasTruthData( ) {

    utility::Matrix meas_noise_cov { models::kMeasNoiseCov };

    // Compute square root of covariance matrix
    utility::MatrixSqrt( meas_noise_cov );

    // Create vector to hold random numbers
    std::vector<float> random_numbers( kSamples * kMeasDim, 0.0f );

    // Create a submatrix for each sample set
    utility::Matrix subset_random_numbers { kMeasDim, 1 };

    // Create matrix to store squared measurement noise covariance matrix times
    // random numbers
    utility::Matrix matrix_mult_result { kMeasDim, 1 };

    // Generate random numbers
    utility::GenerateRandomNum( random_numbers, models::kMeasDistrLowerLimit, models::kMeasDistrUpperLimit );

    // For all sample sets
    for ( int i = 0; i < kSamples; i++ ) {

        // Copy next set of random numbers to matrix data
        for ( int j = 0; j < kMeasDim; j++ ) {
            subset_random_numbers.val[j] = random_numbers[i * kMeasDim + j];
        }

        // Perform matrix multiplication squared measurement noise covariance
        // matrix times
        utility::MatrixMult( matrix_mult_result, meas_noise_cov, subset_random_numbers );

        // Compute measurement function at each time step
        models::MeasModelMath( &system_truth_data_[i * kSysDim], &meas_truth_data_[i * kMeasDim] );

        // Store measurement function + (noise covariance matrix * randon
        // numbers)
        for ( int j = 0; j < kMeasDim; j++ ) {
            meas_truth_data_[i * kMeasDim + j] += matrix_mult_result.val[j];
        }
    }
}

void GenerateData::ReadInTruthData( std::string const &system_data_file,
                                    std::string const &meas_data_file,
                                    int const &        num_mcs ) {

    // Read in truth system data
    ReadInSystemTruthData( system_data_file, num_mcs );

    // Read in truth measurement data
    ReadInMeasurementData( meas_data_file, num_mcs );
}

void GenerateData::ReadInSystemTruthData( std::string const &system_data_file, int const &num_mcs ) {

    ReadDataFromFile( system_data_file, kSysDim, num_mcs, system_truth_data_ );
}

void GenerateData::ReadInMeasurementData( std::string const &meas_data_file, int const &num_mcs ) {

    ReadDataFromFile( meas_data_file, kMeasDim, num_mcs, meas_truth_data_ );
}

void GenerateData::ReadDataFromFile( std::string const & truth_data,
                                     int const &         model_dim,
                                     int const &         current_mc,
                                     std::vector<float> &input_data ) {

    // Open the File
    std::ifstream input_file( truth_data.c_str( ), std::ios_base::in );

    // Check if object is valid
    if ( !input_file ) {
        throw std::runtime_error( "Unable to read input file " + truth_data + ". Try running './filters --create'.\n" );
    }

    std::string line {};
    std::getline( input_file, line );
    std::istringstream iss( line );

    int num_mcs {};
    int dim {};
    int samples {};

    // Read in header data
    iss >> num_mcs >> dim >> samples;

    // Check input file versus application parameters
    if ( num_mcs < kNumMcs ) {
        throw std::runtime_error( "Monte Carlos ( " + std::to_string( num_mcs ) + " ) in " + truth_data +
                                  " are less than requested ( " + std::to_string( kNumMcs ) + " )." );
    } else if ( dim != model_dim ) {
        throw std::runtime_error( "Dimensions ( " + std::to_string( dim ) + " ) in " + truth_data +
                                  " don't match requested ( " + std::to_string( kMeasDim ) + " )." );
    } else if ( kSamples > samples ) {
        throw std::runtime_error( "Samples ( " + std::to_string( samples ) + " ) in " + truth_data +
                                  " are less than requested ( " + std::to_string( kSamples ) + " )." );
    }

    // Skip previous Monte Carlo truth data
    int start { current_mc * samples };
    for ( int j = 0; j < start; ++j ) {
        input_file.ignore( std::numeric_limits<std::streamsize>::max( ), '\n' );
    }

    for ( int i = 0; i < kSamples; i++ ) {
        for ( int j = 0; j < model_dim; j++ ) {
            input_file >> input_data[i * model_dim + j];
        }
    }
}

void GenerateData::WriteGeneratedData( std::string const &truth_data ) {

    std::vector<float>::const_iterator it { system_truth_data_.cbegin( ) };
    utility::WriteToFile( truth_data, kSysDim, kSamples, it );
}

void GenerateData::WriteCreatedTruthData( std::string const &system_data_file, std::string const &meas_data_file ) {

    std::vector<float>::const_iterator itSys { system_truth_data_.cbegin( ) };
    utility::WriteToFile( system_data_file, kSysDim, kSamples, itSys );

    std::vector<float>::const_iterator itMeas { meas_truth_data_.cbegin( ) };
    utility::WriteToFile( meas_data_file, kMeasDim, kSamples, itMeas );
}

} /* namespace filters */
