/**
 * @file utilities.h
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains additional functions used throughout
 * the application.
 *
 * @copyright Copyright (c) 2020
 *
 * @license This project is released under the GNU Public License
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */
#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <algorithm>  // std::remove_if, std::nth_element
#include <cassert>    // assert
#include <cblas.h>    //cblas_sgemm
#include <cmath>      // std::sqrt,
#include <fstream>    // std::ofstream
#include <iterator>   // std::vector<>::const_iterator
#include <lapacke.h>  // LAPACKE_sgetrf, LAPACKE_sgetri
#include <map>        // std::map
#include <numeric>    // std::accumulate, std::inner_product
#include <random>  // std::default_random_engine, std::random_device, std::uniform_real_distribution
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <vector>     // std::vector

/**
 * @brief Contains additional functions used throughout application
 */
namespace utility {

constexpr int kSysDim { 4 };  /**< Number of states in system model */
constexpr int kMeasDim { 2 }; /**< Number of states in measurement model */

/**
 * @enum Processor
 * @brief Enumeration of processor types
 */
enum class Processor : bool { kCpu, kGpu };

using ProcessorMap = std::map<Processor, std::string>;

static const ProcessorMap processor_map = { { Processor::kCpu, "CPU" },
                                            { Processor::kGpu, "GPU" } };

/**
 * @enum Timing
 * @brief Enumeration of timing calculations
 */
enum class Timing : int { kMedian, kMean, kStdDev, kCount };

using TimingMap = std::map<Timing, std::string>;

static const TimingMap timing_map = { { Timing::kMedian, "Median" },
                                      { Timing::kMean, "Mean" },
                                      { Timing::kStdDev, "StdDev" } };

/**
 * @enum Filter
 * @brief Enumeration of possible particle filters
 */
enum class Filter : int { kBootstrap, kCount };

using FilterMap = std::map<Filter, std::string>;

static const FilterMap filters_map = { { Filter::kBootstrap, "Bootstrap" } };

/**
 * @enum Method
 * @brief Enumeration of possible resampling method
 */
enum class Method : int { kSystematic, kStratified, kMetropolisC2, kCount };

using MethodMap = std::map<Method, std::string>;

static const MethodMap method_map = { { Method::kSystematic, "Systematic" },
                                      { Method::kStratified, "Stratified" },
                                      { Method::kMetropolisC2,
                                        "MetropolisC2" } };

/**
 * @struct FilterInfo
 * @brief Structure to store user-specified data and send to filter classes
 */
typedef struct _filterInfo {
    int samples;     /**< Quantity of samples specified from input file */
    int particles;   /**< Quantity of particles specified from input file */
    int num_mcs;     /**< Quantity of Monte Carlos specified from input file */
    int resampling;  /**< Systmatic = 0, Stratified = 1, Metropolis = 2 */
    int filter_type; /**< Type of filter: Bootstrap = 0 */
} FilterInfo;

/**
 * @struct Matrix
 * @brief Structure to store matrix and information
 *
 * @param[in] row Stores quantity of rows
 * @param[in] col Stores quantity of columns
 * @param[in] val Store matrix data in a vector
 */
typedef struct _matrix {

    int                row;
    int                col;
    std::vector<float> val;

    _matrix( const _matrix &a ) : row( a.row ), col( a.col ), val( a.val ) {};

    _matrix( int const &                         row,
             int const &                         col,
             std::initializer_list<float> const &list ) :
        row( row ),
        col( col ), val( list ) {}

    _matrix( int const &row, int const &col ) : row( row ), col( col ) {
        val.resize( row * col, 0.0f );
    }

    _matrix( int const &row, int const &col, std::vector<float> const val ) :
        row( row ), col( col ), val( val ) {}
} Matrix;

/**
 * @brief Template function to compute square root of a matrix using floats
 *
 * @see http://www.netlib.org/lapack/lapacke.html
 * @see
 * http://www.netlib.org/lapack/explore-html/d7/d75/lapacke__ssyevr_8c_source.html
 * @see http://www.openblas.net/
 * @see http://www.seehuhn.de/pages/matrixfn
 *
 * @param[in,out] matrix_a Matrix to process
 */
inline void MatrixSqrt( utility::Matrix &matrix_a ) {

    int    n { matrix_a.row };
    float *a { matrix_a.val.data( ) };

    /* allocate space for the output parameters and workspace arrays */
    int   m {};
    float abstol { -1.0f };

    std::vector<float> w( n, 0.0f );
    std::vector<float> z( n * n, 0.0f );
    std::vector<int>   suppz( 2 * n, 0 );

    /* get the eigenvalues and eigenvectors */
    int info = LAPACKE_ssyevr( LAPACK_ROW_MAJOR,
                               'V',
                               'A',
                               'L',
                               n,
                               a,
                               n,
                               0,
                               0,
                               0,
                               0,
                               abstol,
                               &m,
                               w.data( ),
                               z.data( ),
                               n,
                               suppz.data( ) );

    if ( info != 0 ) {
        throw std::runtime_error( "Issue with LAPACKE_ssyevr.\n" );
    }

    /* allocate and Initialize a new matrix B=Z*D */
    std::vector<float> b( n * n, 0.0f );
    for ( int j = 0; j < n; ++j ) {
        float lambda = std::sqrt( w[j] );
        for ( int i = 0; i < n; ++i ) {
            b[i * n + j] = z[i * n + j] * lambda;
        }
    }

    float alpha { 1.0f };
    float beta { 0.0f };

    /* calculate the square root A=B*Z^T */
    cblas_sgemm( CblasRowMajor,
                 CblasNoTrans,
                 CblasTrans,
                 n,
                 n,
                 n,
                 alpha,
                 b.data( ),
                 n,
                 z.data( ),
                 n,
                 beta,
                 a,
                 n );
}

/**
 * @brief Matrix-Matrix multiplication function using floats
 *
 * @see http://www.openblas.net/
 *
 * @param[out] matrix_c Structure to store A * B
 * @param[in] matrix_a Structure reference to matrix A
 * @param[in] matrix_b Structure reference to matrix B
 */

inline void MatrixMult( utility::Matrix &      matrix_c,
                        utility::Matrix const &matrix_a,
                        utility::Matrix const &matrix_b ) {

    assert( matrix_a.col == matrix_b.row );

    float const *a { matrix_a.val.data( ) };
    float const *b { matrix_b.val.data( ) };
    float *      c { matrix_c.val.data( ) };

    float alpha { 1.0f };
    float beta { 0.0f };

    cblas_sgemm( CblasRowMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 matrix_a.row,
                 matrix_b.col,
                 matrix_b.row,
                 alpha,
                 a,
                 matrix_b.row,
                 b,
                 matrix_b.col,
                 beta,
                 c,
                 matrix_b.col );
}

/**
 * @brief Matrix-Matrix-Matrix multiplication function using floats
 *
 * @see http://www.openblas.net/
 *
 * @param[out] matrix_d Structure to store A * B * C
 * @param[in] matrix_a Structure reference to matrix A
 * @param[in] matrix_b Structure reference to matrix B
 * @param[in] matrix_c Structure reference to matrix C
 */
inline void MatrixMult( utility::Matrix &      matrix_d,
                        utility::Matrix const &matrix_a,
                        utility::Matrix const &matrix_b,
                        utility::Matrix const &matrix_c ) {

    assert( matrix_a.col == matrix_b.row );
    assert( matrix_b.col == matrix_c.row );

    float const *a { matrix_a.val.data( ) };
    float const *b { matrix_b.val.data( ) };
    float const *c { matrix_c.val.data( ) };
    float *      d { matrix_d.val.data( ) };

    std::vector<float> temp_storage( matrix_a.row * matrix_b.col, 0.0f );
    float *            ab { temp_storage.data( ) };  // Temp store for A*B

    float alpha { 1.0f };
    float beta { 0.0f };

    cblas_sgemm( CblasRowMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 matrix_a.row,
                 matrix_b.col,
                 matrix_b.row,
                 alpha,
                 a,
                 matrix_b.row,
                 b,
                 matrix_b.col,
                 beta,
                 ab,
                 matrix_b.col );

    cblas_sgemm( CblasRowMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 matrix_a.row,
                 matrix_c.col,
                 matrix_c.row,
                 alpha,
                 ab,
                 matrix_c.row,
                 c,
                 matrix_c.col,
                 beta,
                 d,
                 matrix_c.col );
}

/**
 * @brief Matrix-Vector multiplication function using floats
 *
 * @see http://www.openblas.net/
 *
 * @param[out] vector_c Vector to store A * B
 * @param[in] matrix_a Structure reference to matrix A
 * @param[in] vector_b Vector reference to vector B
 */
inline void MatrixMult( std::vector<float> &      vector_c,
                        utility::Matrix const &   matrix_a,
                        std::vector<float> const &vector_b ) {

    assert( matrix_a.col == vector_b.size( ) );

    float const *a { matrix_a.val.data( ) };
    float const *b { vector_b.data( ) };
    float *      c { vector_c.data( ) };

    float alpha { 1.0f };
    float beta { 0.0f };

    cblas_sgemv( CblasRowMajor,
                 CblasNoTrans,
                 matrix_a.row,
                 matrix_a.col,
                 alpha,
                 a,
                 matrix_a.col,
                 b,
                 1,
                 beta,
                 c,
                 1 );
}

/**
 * @brief Function to compute the inversion of a matrix using floats
 *
 * @see http://www.netlib.org/lapack/lapacke.html
 * @see
 * http://www.netlib.org/lapack/explore-html/d2/d96/lapacke__dgetrf_8c_source.html
 * @see
 * http://www.netlib.org/lapack/explore-html/da/d0e/lapacke__dgetri_8c_source.html
 * @see
 * https://stackoverflow.com/questions/3519959/computing-the-inverse-of-a-matrix-using-lapack-in-c
 *
 * @param[in,out] matrix_a Matrix to be inverted
 */
inline void ComputeInverse( utility::Matrix &matrix_a ) {

    // A must be a square matrix???
    int              n { matrix_a.row };
    float *          a = matrix_a.val.data( );
    std::vector<int> piv( n + 1, 0 );

    int info {};
    info = LAPACKE_sgetrf( LAPACK_ROW_MAJOR, n, n, a, n, piv.data( ) );

    if ( info != 0 ) {
        throw std::runtime_error( "Issue with LAPACKE_sgetrf.\n" );
    }

    info = LAPACKE_sgetri( LAPACK_ROW_MAJOR, n, a, n, piv.data( ) );

    if ( info != 0 ) {
        throw std::runtime_error( "Issue with LAPACKE_sgetri.\n" );
    }
}

/**
 * @brief Normal distribution random number generator
 *
 * @param[in] random_numbers Stores random numbers
 * @param[in] lBound Lower bound of distribution
 * @param[in] uBound Upper bound of distribution
 */
inline void GenerateRandomNum( std::vector<float> &random_numbers,
                               float const &       lBound,
                               float const &       uBound ) {

    // Random device class instance, source of 'true' randomness for
    // initializing random seed
    std::default_random_engine gen( std::random_device {}( ) );

    // Instance of class std::normal_distribution with specific mean and stddev
    // Lower and Upper limits are declared in base class
    std::normal_distribution<float> distr( lBound, uBound );

    // Compute random numbers for all particles
    for_each( random_numbers.begin( ),
              random_numbers.end( ),
              [&distr, &gen]( float &a ) { a = distr( gen ); } );
}

/**
 * @brief Compute median for Monte Carlos times
 *
 * @param[in/out] time_vector Vector holding times for all Monte Carlos
 * @param[out] median Median time for all Monte Carlos
 */
inline void ComputeMedianTime( std::vector<float> &time_vector,
                               float &             median ) {
    // Compute median
    std::nth_element( time_vector.begin( ),
                      time_vector.begin( ) + time_vector.size( ) / 2,
                      time_vector.end( ) );
    median = time_vector[time_vector.size( ) / 2];

    // Remove outliers
    time_vector.erase( std::remove_if( time_vector.begin( ),
                                       time_vector.end( ),
                                       [&median]( float const &n ) {
                                           return ( n > ( median * 1.05f ) );
                                       } ),
                       time_vector.end( ) );
}

/**
 * @brief Compute mean for Monte Carlos times
 *
 * @param[in] time_vector Vector holding times for all Monte Carlos
 * @param[out] mean Mean time for all Monte Carlos
 */
inline void ComputeMeanTime( std::vector<float> const &time_vector,
                             float &                   mean ) {
    // Compute mean
    mean = std::accumulate( time_vector.begin( ), time_vector.end( ), 0.0f ) /
           time_vector.size( );
}

/**
 * @brief Compute standard deviation for Monte Carlos times
 *
 * @param[in] time_vector Vector holding times for all Monte Carlos
 * @param[in] mean Mean time for all Monte Carlos
 * @param[out] stdDev
 */
inline void ComputeStdDevTime( std::vector<float> const &time_vector,
                               float const &             mean,
                               float &                   stdDev ) {

    // Compute standard deviation
    std::vector<float> diff( time_vector.size( ) );
    std::transform( time_vector.begin( ),
                    time_vector.end( ),
                    diff.begin( ),
                    [&mean]( float const &x ) { return ( x - mean ); } );

    // Compute difference
    float sqrtSum { std::inner_product(
        diff.begin( ), diff.end( ), diff.begin( ), 0.0f ) };  // Compute product
    stdDev = std::sqrt( sqrtSum / time_vector.size( ) );
}

/**
 * @brief Compute median, mean, standard deviation for Monte Carlo times
 *
 * @param[in/out] time_vector
 * @param[in/out] median
 * @param[in/out] mean
 * @param[in/out] stdDev
 */
inline void ComputeOverallTiming( std::vector<float> &timeVec,
                                  float &             median,
                                  float &             mean,
                                  float &             stdDev ) {
    ComputeMedianTime( timeVec, median );
    ComputeMeanTime( timeVec, mean );
    ComputeStdDevTime( timeVec, mean, stdDev );
}

/**
 * @brief Compute overall timing across all Monte Carlos
 *
 * @param[in] timing_results 2D vector containing timing results for each Monte
 * Carlo
 */
inline void
PrintTimingResults( std::vector<std::vector<float>> const &timing_results ) {

    for ( int i = 0; i < static_cast<int>( utility::Timing::kCount ); i++ ) {
        std::printf(
            "%s\t ",
            utility::timing_map.find( utility::Timing( i ) )->second.c_str( ) );
    }
    std::printf( "\n" );

    for ( int i = 0; i < static_cast<int>( utility::Timing::kCount ); i++ ) {
        float sum { std::accumulate(
            timing_results[i].begin( ), timing_results[i].end( ), 0.0f ) };
        float mean { sum / timing_results[i].size( ) };
        std::printf( "%0.0f\t ", mean );
    }
    std::printf( "\n\n" );
}

/**
 * @brief Writes data to output file for generated system
 *        and measurement truth data
 *
 * @param[in] filename Output filename
 * @param[in] dim Number of system dimensions
 * @param[in] samples Number of samples
 * @param[in] it Iterator to beginning of vector to be printed
 */
inline void WriteToFile( std::string const &                          filename,
                         int const &                                  dim,
                         int const &                                  samples,
                         typename std::vector<float>::const_iterator &it ) {

    // We jump to the end to add each Monte Carlo
    std::ofstream fout( filename, std::ios_base::app );
    fout.precision( 15 );
    fout.setf( std::ios::fixed, std::ios::floatfield );
    // fout << dim << " " << samples << "\n";
    for ( int i = 0; i < samples; i++ ) {
        int idxS { i * dim };
        int idxE { idxS + dim };
        std::copy(
            it + idxS, it + idxE, std::ostream_iterator<float>( fout, " " ) );
        fout << "\n";
    }
    fout.close( );
}

/**
 * @brief Writes data to output file for filtered estimates
 *
 * @param[in] filename Output filename
 * @param[in] type Filter type
 * @param[in] samples Number of samples
 * @param[in] median Timing median across all Monte Carlos
 * @param[in] mean Timing mean across all Monte Carlos
 * @param[in] stdDev Timing standard deviation across all Monte Carlos
 * @param[in] it Iterator to beginning of vector to be printed
 */
inline void WriteToFile( std::string const &                          filename,
                         std::string const &                          type,
                         int const &                                  samples,
                         float const &                                median,
                         float const &                                mean,
                         float const &                                stdDev,
                         typename std::vector<float>::const_iterator &it ) {

    // We jump to the end to add each Monte Carlo
    std::ofstream fout( filename, std::ios_base::app );
    fout.setf( std::ios::fixed, std::ios::floatfield );
    fout << type << " " << median << " " << mean << " " << stdDev << "\n";
    fout.precision( 15 );
    for ( int i = 0; i < samples; i++ ) {
        int idxS { i * utility::kSysDim };
        int idxE { idxS + utility::kSysDim };
        std::copy(
            it + idxS, it + idxE, std::ostream_iterator<float>( fout, " " ) );
        fout << "\n";
    }
    fout.close( );
}

/**
 * @brief Write header to output file for generated system
 *        and measurement truth data
 *
 * @param[in] filename Output filename
 * @param[in] info Struct containing filter information
 */
inline void WriteTruthHeader( std::string const &filename,
                              int const &        num_mcs,
                              int const &        dim,
                              int const &        samples ) {
    std::ofstream fout( filename, std::ios_base::trunc );
    fout.setf( std::ios::fixed, std::ios::floatfield );
    fout << num_mcs << " " << dim << " " << samples << "\n";
    fout.close( );
}

/**
 * @brief Write header to output file.
 *
 * @param[in] filename Output filename
 * @param[in] info Struct containing filter information
 */
inline void WriteDataHeader( std::string const &        filename,
                             utility::FilterInfo const &filter_info ) {
    std::ofstream fout( filename, std::ios_base::trunc );
    fout.setf( std::ios::fixed, std::ios::floatfield );
    fout << filter_info.num_mcs << " " << kSysDim << " " << filter_info.samples
         << " " << filter_info.particles << "\n";
    fout.close( );
}

#ifdef USE_NVTX
#include "nvtx3/nvToolsExt.h"

uint32_t const colors[] { 0x0000ff00, 0x0f0000ff, 0x000000ff,
                          0x00ffff00, 0x00ff00ff, 0x0000ffff,
                          0x00ff0000, 0x00ffffff, 0x0f0f0f0f };
int const      num_colors { sizeof( colors ) / sizeof( uint32_t ) };

/**
 * class Tracer
 */
class Tracer {
  public:
    /**
     * @brief Construct a new Tracer object
     *
     * @see
     * https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm
     * @see
     * https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
     *
     * @param[in] name Function name
     * @param[in] cid color id
     */

    Tracer( char const *name, int const &cid ) {
        int color_id                      = cid;
        color_id                          = color_id % num_colors;
        nvtxEventAttributes_t eventAttrib = {};
        eventAttrib.version               = NVTX_VERSION;
        eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType             = NVTX_COLOR_ARGB;
        eventAttrib.color                 = colors[color_id];
        eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii         = name;
        nvtxRangePushEx( &eventAttrib );
    }

    /**
     * @brief Destroy the Tracer object
     *
     */
    ~Tracer( ) {
        nvtxRangePop( );
    }
};

#define RANGE( name, cid ) utility::Tracer uniq_name_using_macros( name, cid );
#else
#define RANGE( name, cid )
#endif

} /* namespace utility */

#endif /* UTILITIES_H_ */
