/**
 * @file generate_data.h
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains header information for GenerateData class.
 *
 * @copyright Copyright (c) 2020
 *
 * @license This project is released under the GNU Public License
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */

#ifndef GENERATE_DATA_H_
#define GENERATE_DATA_H_

#include <string>  // std::string
#include <vector>  // std::vector

#include "utilities.h"

/**
 * @brief Contains functions for filter functionality
 *
 */
namespace filters {

/**
 * @class GenerateData
 * @brief Generate input data
 */
class GenerateData {
  public:
    /**
     * @brief Construct a new GenerateData object
     *
     */
    GenerateData( ) noexcept = default;
    /**
     * @brief Construct a new GenerateData object
     *
     * @param[in] info Structure containing information about filter
     */
    explicit GenerateData( utility::FilterInfo const &filter_info );

    /**
     * @brief Destroy the GenerateData object
     *
     */
    virtual ~GenerateData( ) noexcept;

    /**
     * @brief Copy construct a new GenerateData object
     *
     */
    GenerateData( GenerateData const & ) = default;

    /**
     * @brief Copy assignment construct a new GenerateData object
     *
     */
    GenerateData &operator=( GenerateData const & ) = delete;

    /**
     * @brief Move construct a new GenerateData object
     *
     */
    GenerateData( GenerateData && ) noexcept = default;

    /**
     * @brief Move assignment construct a new GenerateData object
     *
     */
    GenerateData &operator=( GenerateData && ) noexcept = delete;

    /**
     * @brief Get true measurement for the current time step
     *
     * @param[in] idx Index to true measurement for the current time step
     * @param[out] result Latest measurement data
     */
    void get_current_meas_data( int const &idx, float *current_meas_data );

    /**
     * @brief Create a truth data to be stored and used later
     *
     * @param[in] system_data_file System truth file
     * @param[in] meas_data_file Measurement truth file
     *
     * Calls the following
     * + generatedData
     * + WriteCreatedTruthData
     */
    void CreateTruthData( std::string const &system_data_file,
                          std::string const &meas_data_file );

    /**
     * @brief Generate truth data for each Monte Carlo in real-time
     *
     * Calls the following
     * + GenerateSystemTruthData
     * + GenerateMeasTruthData
     */
    void GenerateTruthData( );

    /**
     * @brief Read in input data from file
     *
     * @param[in] system_data_file File containing system state data
     * @param[in] meas_data_file File containing measurement state data
     * @param[in] num_mcs Monte Carlo to start on
     *
     * Calls the following
     * + ReadInSystemTruthData
     * + ReadInMeasurementData
     */
    void ReadInTruthData( std::string const &system_data_file,
                          std::string const &meas_data_file,
                          int const &        num_mcs );

    /**
     * @brief Write generated data to file
     *
     * @param[in] input_file Filename to store data
     */
    void WriteGeneratedData( std::string const &truth_data );

    /**
     * @brief Write system and measurement truth data to files
     *
     * @param[in] system_data_file File containing system state data
     * @param[in] meas_data_file File containing measurement state data
     */
    void WriteCreatedTruthData( std::string const &system_data_file,
                                std::string const &meas_data_file );

  private:
    /**
     * @brief Generate truth data for system model
     *
     */
    void GenerateSystemTruthData( );

    /**
     * @brief Generate truth data for measurement model
     */
    void GenerateMeasTruthData( );

    /**
     * @brief Read in system data from input file
     * @param[in] system_data_file File containing system state data
     * @param[in] num_mcs Monte Carlo to start on
     *
     * Calls the following
     * + ReadDataFromFile
     */
    void ReadInSystemTruthData( std::string const &system_data_file,
                                int const &        num_mcs );

    /**
     * @brief Read in measurement data from input file
     * @param[in] meas_data_file File containing measurement state data
     * @param[in] num_mcs Monte Carlo to start on
     *
     * Calls the following
     * + ReadDataFromFile
     */
    void ReadInMeasurementData( std::string const &meas_data_file,
                                int const &        num_mcs );

    /**
     * @brief Construct a new read Data From File object
     *
     * @param[in] filename File to read truth data from
     * @param[in] model_dim Dimension of data to be read in
     * @param[in] num_mcs Monte Carlo being executed
     * @param[out] input_data Input data to be used in application
     */
    void ReadDataFromFile( std::string const & filename,
                           int const &         model_dim,
                           int const &         current_mc,
                           std::vector<float> &input_data );

    int const kSysDim;  /**< The quantity of dimensions of the system */
    int const kMeasDim; /**< The quantity of dimensions of the measurement */
    int const kSamples; /**< The quantity of samples to be processed */
    int const kNumMcs;  /**< The quantity of Monte Carlos */

    std::vector<float> system_truth_data_; /**< Vector to store actual data :
                                              system dimension * samples */
    std::vector<float> meas_truth_data_; /**< Vector to store true measurements
                                            : measurement dimension * samples */
};

} /* namespace filters */

#endif /* GENERATE_DATA_H_ */
