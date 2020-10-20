/**
 * @file filters.cpp
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains the main program
 *
 * @copyright Copyright (c) 2020
 *
 * @license This project is released under the GNU Public License
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */

#include <algorithm>   // std::remove
#include <functional>  // std::bind, std::placeholders
#include <stdexcept>   // std::runtime_error
#include <string>      // std::string, std::to_string
#include <sys/stat.h>  // mkdir
#include <vector>      // std::vector

#include "command_line_options.h"
#include "generate_data.h"
#include "particle_bpf_cpu.h"
#include "particle_bpf_gpu.h"

int main( int const argc, char const *argv[] ) {

    std::string const kDir { "./data/" };
    std::string const kSysTruthFile { kDir + "SysTruthData.txt" };
    std::string const kMeasTruthFile { kDir + "MeasTruthData.txt" };

    // Make directory if is doesn't exist
    if ( mkdir( kDir.c_str( ), 0777 ) == 0 ) {
        std::printf( "Successfully created %s directory\n", kDir.c_str( ) );
    };

    // Get user inputs
    command_line::CommandLineOptions readCL {};

    readCL.ParseInputs( argc, argv );

    // Create static truth data, exit program if successful
    if ( readCL.get_create_data_flag( ) ) {
        filters::GenerateData gd( readCL.get_filter_info( ) );
        gd.CreateTruthData( kSysTruthFile, kMeasTruthFile );
    }

    std::string estimate_file {};
    std::string truth_file {};

    // Create input and output files
    truth_file = kDir + "truth_" +
                 utility::processor_map.find( utility::Processor( readCL.get_use_gpu_flag( ) ) )->second + "_" +
                 utility::method_map.find( utility::Method( readCL.get_resampling( ) ) )->second + "_" +
                 std::to_string( readCL.get_particles( ) ) + ".txt";

    utility::WriteDataHeader( truth_file, readCL.get_filter_info( ) );

    estimate_file = kDir + "estimate_" +
                    utility::processor_map.find( utility::Processor( readCL.get_use_gpu_flag( ) ) )->second + "_" +
                    utility::method_map.find( utility::Method( readCL.get_resampling( ) ) )->second + "_" +
                    std::to_string( readCL.get_particles( ) ) + ".txt";

    utility::WriteDataHeader( estimate_file, readCL.get_filter_info( ) );

    // Store average timing results for all Monte Carlos
    std::vector<std::vector<float>> timing_results( static_cast<int>( utility::Timing::kCount ),
                                                    std::vector<float>( readCL.get_num_mcs( ), 0.0f ) );

    for ( int mc = 0; mc < readCL.get_num_mcs( ); mc++ ) {

        filters::GenerateData gd( readCL.get_filter_info( ) );

        if ( readCL.get_use_truth_flag( ) ) {
            // Read from precalculate truth files
            gd.ReadInTruthData( kSysTruthFile, kMeasTruthFile, mc );
        } else {
            // Generate data each Monte Carlo
            gd.GenerateTruthData( );
        }

        // Write generated data to file to be used for RMSE analysis later
        gd.WriteGeneratedData( truth_file );

        // Function pointer to generated measurement data
        auto truth_meas_func_ptr = std::bind(
            &filters::GenerateData::get_current_meas_data, gd, std::placeholders::_1, std::placeholders::_2 );

        // If bootstrap particle filter
        if ( readCL.get_filter_type( ) == static_cast<int>( utility::Filter::kBootstrap ) ) {
            // Use GPU version
            if ( readCL.get_use_gpu_flag( ) ) {
                filters::ParticleBpfGpu<float> bpf( readCL.get_filter_info( ), truth_meas_func_ptr );
                bpf.Initialize( mc, timing_results );
                bpf.WriteOutput( estimate_file );
            } else {  // Use CPU version
                filters::ParticleBpfCpu bpf( readCL.get_filter_info( ), truth_meas_func_ptr );
                bpf.Initialize( mc, timing_results );
                bpf.WriteOutput( estimate_file );
            }
        }
    }

    // Print average timing
    std::printf( "%s: %s: Monte Carlos %d: Samples %d: Particles: %d\n",
                 utility::processor_map.find( utility::Processor( readCL.get_use_gpu_flag( ) ) )->second.c_str( ),
                 utility::method_map.find( utility::Method( readCL.get_resampling( ) ) )->second.c_str( ),
                 readCL.get_num_mcs( ),
                 readCL.get_samples( ),
                 readCL.get_particles( ) );
    std::printf( "Data stored in %s and %s\n", estimate_file.c_str( ), truth_file.c_str( ) );
    std::printf( "Average Times (us)\n" );
    utility::PrintTimingResults( timing_results );

    return ( EXIT_SUCCESS );
}
