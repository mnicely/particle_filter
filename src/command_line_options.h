/**
 * @file command_line_options.h
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains header information for CommandLineOptions class.
 *
 * @copyright Copyright (c) 2020
 *
 * @license This project is released under the GNU Public License
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */

#ifndef COMMAND_LINE_OPTIONS_H_
#define COMMAND_LINE_OPTIONS_H_

#include <boost/program_options/options_description.hpp>  // boost::program_options::options_description
#include <string>                                         // std::string

#include "utilities.h"

/**
 * @brief Contains functions to parse command line inputs
 */
namespace command_line {

/**
 * @class CommandLineOptions
 * @brief Contains implementation to parse command line arguments
 */
class CommandLineOptions {
  public:
    /**
     * @brief Construct a new CommandLineOptions object
     *
     */
    CommandLineOptions( ) noexcept;

    /**
     * @brief Destroy the CommandLineOptions object
     *
     */
    virtual ~CommandLineOptions( ) noexcept;

    /**
     * @brief Copy construct a new CommandLineOptions object
     *
     */
    CommandLineOptions( CommandLineOptions const & ) = delete;

    /**
     * @brief Copy assignment construct a new CommandLineOptions object
     *
     */
    CommandLineOptions &operator=( CommandLineOptions const & ) = delete;

    /**
     * @brief Move construct a new CommandLineOptions object
     *
     */
    CommandLineOptions( CommandLineOptions && ) noexcept = delete;

    /**
     * @brief Move assignment construct a new CommandLineOptions object
     *
     */
    CommandLineOptions &operator=( CommandLineOptions && ) noexcept = delete;

    /**
     * @brief Parse command line inputs

     * @param[in] argc An integer argument count of the command line arguments.
     * @param[in] argv An argument vector of the command line arguments.
     */
    void ParseInputs( int const argc, char const *argv[] );

    /**
     * @brief Get the filter type
     *
     * @return filter_info_.filter_type
     */
    int const &get_filter_type( ) const;

    /**
     * @brief Get the resampling method type
     *
     * @return filter_info_.resampling
     */
    int const &get_resampling( ) const;

    /**
     * @brief Get the number of particles
     *
     * @return filter_info_.particles
     */
    int const &get_particles( ) const;

    /**
     * @brief Get the number of samples
     *
     * @return filter_info_.samples
     */
    int const &get_samples( ) const;

    /**
     * @brief Get the number of Monte Carlos
     *
     * @return filter_info_.mcs
     */
    int const &get_num_mcs( ) const;

    /**
     * @brief Get decision whether to create truth data
     *
     * @return create_data_flag_
     */
    bool const &get_create_data_flag( ) const;
    /**
     * @brief Get decision whether to execute on GPU
     *
     * @return use_gpu_flag_
     */
    bool const &get_use_gpu_flag( ) const;

    /**
     * @brief Get decision whether to use truth data
     *
     * @return use_truth_flag_
     */
    bool const &get_use_truth_flag( ) const;

    /**
     * @brief Get structure containing information about filter
     *
     * @return filter_info_
     */
    utility::FilterInfo const &get_filter_info( ) const;

  private:
    utility::FilterInfo                         filter_info_;      /**< User-specified arguments */
    boost::program_options::options_description my_options_;       /**< Store cli */
    bool                                        create_data_flag_; /**< If true, create system and measurement truth */
    bool                                        use_truth_flag_;   /**< If true, use truth data */
    bool                                        use_gpu_flag_;     /**< If true, use GPU for analysis */
};

}  // namespace command_line

#endif /* COMMAND_LINE_OPTIONS_H_ */
