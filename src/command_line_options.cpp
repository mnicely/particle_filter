/**
 * @file command_line_options.cpp
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains implementation to parse command line arguments.
 * This utility is developed using the Boost library Program_Options.
 * It will parse command line argument to find input file and chosen
 * filters and number of particles per filter.

 * @copyright Copyright (c) 2020
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */

#include <boost/program_options/options_description.hpp>  // boost::program_options::options_description
#include <boost/program_options/parsers.hpp>  // boost::program_options::parse_command_line
#include <boost/program_options/value_semantic.hpp>  // boost::program_options::bool_switch, boost::program_options::value
#include <boost/program_options/variables_map.hpp>  // boost::program_options::variables_map, boost::program_options::store, boost::program_options::notify
#include <exception>                                // std::exception
#include <iostream>                                 // std::cout
#include <stdexcept>                                // std::runtime_error

#include "command_line_options.h"

namespace command_line {

CommandLineOptions::CommandLineOptions( ) noexcept :
    // clang-format off
    filter_info_ {},
    my_options_ {}, 
    create_data_flag_ {},
    use_truth_flag_ {}, 
    use_gpu_flag_ {}
{
    boost::program_options::options_description options( "Program Options", 120 );
    options.add_options( )
        // clang-format off
            ( "filter,f", boost::program_options::value<int>( &filter_info_.filter_type )->default_value( 0 ), "Type of filter: Bootstrap = 0" )
    		( "particles,p", boost::program_options::value<int>( &filter_info_.particles )->default_value( 65536 ), "Number of particles" )
            ( "samples,s", boost::program_options::value<int>( &filter_info_.samples )->default_value( 500 ), "Number of samples to execute" )
            ( "resampling,r", boost::program_options::value<int>( &filter_info_.resampling )->default_value( 0 ), "Resampling method: Systmatic = 0, Stratified = 1, Metropolis = 2" )
            ( "mcs,m", boost::program_options::value<int>( &filter_info_.num_mcs )->default_value( 5 ), "Number of Monte Carlos to execute" )
            ( "create,c", boost::program_options::bool_switch( &create_data_flag_ )->default_value( false ), "Create truth data" )   
            ( "truth,t", boost::program_options::bool_switch( &use_truth_flag_ )->default_value( false ), "Use precalculate truth data" )  
            ( "gpu,g", boost::program_options::bool_switch( &use_gpu_flag_ )->default_value( false ), "Use GPU or CPU" )
            ( "help,h", "Display help menu." );
    // clang-format on
    my_options_.add( options );
}

CommandLineOptions::~CommandLineOptions( ) noexcept {}

void CommandLineOptions::ParseInputs( int const argc, char const *argv[] ) {
    boost::program_options::variables_map var_map {};

    try {
        boost::program_options::store(
            boost::program_options::parse_command_line(
                argc, argv, my_options_ ),
            var_map );
        boost::program_options::notify( var_map );

        if ( var_map.count( "help" ) ) {
            std::cout << my_options_ << std::endl;
            exit( EXIT_FAILURE );
        }

    } catch ( std::exception &e ) {
        throw std::runtime_error( "Parsing error - " +
                                  std::string( e.what( ) ) );

    } catch ( ... ) {
        throw std::runtime_error(
            "Parsing error (unknown type)!\n Run ./Filters -h for help" );
    }

    // Check sample parameter
    if ( filter_info_.samples <= 0 ) {
        filter_info_.samples = 500;
        std::printf( "Number of samples must be a greater than 0\n" );
        std::printf( "Number of samples increased to %d\n\n",
                     filter_info_.samples );
    }

    // Check particle parameter
    if ( ( filter_info_.particles % 128 != 0 ) && use_gpu_flag_ ) {

        filter_info_.particles =
            ceil( static_cast<float>( filter_info_.particles ) / 128 ) * 128;
        std::printf( "Number of particles must be a multiple of 128\n" );
        std::printf( "Number of particles increased to %d\n\n",
                     filter_info_.particles );

    } else if ( filter_info_.particles <= 0 ) {

        filter_info_.particles = 65536;
        std::printf( "Number of particles must be greater than 0\n" );
        std::printf( "Number of particles changed to %d\n\n",
                     filter_info_.particles );
    }

    // Check resampling method parameter
    if ( filter_info_.resampling >=
         static_cast<int>( utility::Method::kCount ) ) {

        std::printf( "Resampling method options: \n" );
        for ( int i = 0; i < static_cast<int>( utility::Method::kCount );
              i++ ) {
            std::printf( "%d -> %s \n",
                         i,
                         utility::method_map.find( utility::Method( i ) )
                             ->second.c_str( ) );
        }

        throw std::runtime_error(
            "Incorrect resampling method!\n Run ./Filters -h for help" );

    } else if ( filter_info_.resampling ==
                    static_cast<int>( utility::Method::kMetropolisC2 ) &&
                ( !use_gpu_flag_ ) ) {
        throw std::runtime_error( "CPU version of Metropolis not "
                                  "implemented!\n Run ./Filters -h for help" );
    }

    // Check number of Monte Carlos
    if ( !filter_info_.num_mcs ) {
        filter_info_.num_mcs = 10;
        std::printf( "Number of Monte Carlos must be greater than 0\n" );
        std::printf( "Number of Monte Carlos increased to %d\n\n",
                     filter_info_.num_mcs );
    }
}

int const &CommandLineOptions::get_filter_type( ) const {
    return ( filter_info_.filter_type );
}

int const &CommandLineOptions::get_resampling( ) const {
    return ( filter_info_.resampling );
}

int const &CommandLineOptions::get_particles( ) const {
    return ( filter_info_.particles );
}

int const &CommandLineOptions::get_samples( ) const {
    return ( filter_info_.samples );
}

int const &CommandLineOptions::get_num_mcs( ) const {
    return ( filter_info_.num_mcs );
}

bool const &CommandLineOptions::get_create_data_flag( ) const {
    return ( create_data_flag_ );
}

bool const &CommandLineOptions::get_use_gpu_flag( ) const {
    return ( use_gpu_flag_ );
}

bool const &CommandLineOptions::get_use_truth_flag( ) const {
    return ( use_truth_flag_ );
}

utility::FilterInfo const &CommandLineOptions::get_filter_info( ) const {
    return ( filter_info_ );
}

}  // namespace command_line
