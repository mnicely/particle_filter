/**
 * @file models.h
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains system and measurement models, along with initialization
 * and covariance matrices.
 *
 * @copyright Copyright (c) 2020
 *
 * @license This project is released under the GNU Public License
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */
#ifndef MODELS_H_
#define MODELS_H_

#include <cmath>  // std::atan

#include <cuda_runtime.h>  // __host__, __device__

#include "utilities.h"

/**
 * @brief Contains system and measurements models and covariance matrices
 *
 */
namespace models {

/**
 * @brief Template to compute signum
 *
 * @see https://en.wikipedia.org/wiki/Sign_function
 *
 * @param[in] x Value from function (float or double)
 * @return -1 if x < 0
 * @return 0 if x = 0
 * @return 1 if x > 0
 */

inline __host__ __device__ int Signum( float const &x ) {
    return ( x < -std::numeric_limits<float>::epsilon( )
                 ? -1
                 : x > std::numeric_limits<float>::epsilon( ) );
}

/**
 * @brief System model with 4 dimensions
 *
 * @see http://user.it.uu.se/~thosc112/pubpdf/schongn2005.pdf
 *
 * @param[in] input Vector with particle state data
 * @param[out] system_model_result Vector with updated particle state datA
 */
inline __host__ __device__ void SysModelMath( float const *input,
                                              float *system_result ) {
    /*
     * Computes p(x_t | x_(t-1)
     *
     * [atan(x(1,:)) + 	x(2,:);
     *					x(2,:) 	+	0.3*x(3,:);
     *								0.92*x(3,:) - 	0.3*x(4,:);
     *								0.3*x(3,:) 	+ 	0.92*x(4,:)]
     */

    int idx {};

    system_result[0] = std::atan( input[idx] ) + input[idx + 1];
    system_result[1] = input[idx + 1] + 0.3f * input[idx + 2];
    system_result[2] = 0.92f * input[idx + 2] - 0.3f * input[idx + 3];
    system_result[3] = 0.3f * input[idx + 2] + 0.92f * input[idx + 3];
}

/**
 * @brief Measurement model with 2 dimensions
 *
 * @see http://user.it.uu.se/~thosc112/pubpdf/schongn2005.pdf
 *
 * @param[in] input Vector with measurement data
 * @param[out] meas_model_result Vector with updated measurement datA
 */
inline __host__ __device__ void MeasModelMath( float const *input,
                                               float *meas_result ) {
    /*
     * Computes p(y | x_t)
     *
     * [(0.1f * x(1,:).^2) .* sign(x(1,:));
     * 										x(2,:) - x(3,:) + x(4,:)]
     */
    int idx {};

    meas_result[0] = 0.1f * ( input[idx] * input[idx] ) * Signum( input[idx] );
    meas_result[1] = input[idx + 1] - input[idx + 2] + input[idx + 3];
}

static float const kSysDistrLowerLimit {
    0.0f
}; /**< System model's lower distribution limit */
static float const kSysDistrUpperLimit {
    1.0f
}; /**< System model's upper distribution limit */
static float const kMeasDistrLowerLimit {
    0.0f
}; /**< Measurement model's lower distribution limit */
static float const kMeasDistrUpperLimit {
    0.1f
}; /**< Measurement model's upper distribution limit */

// clang-format off

// *************************************** Model data for 4D ***********************************************
/**
 * @brief Structure to store initial states : system dimension * 1
 *
 */
static utility::Matrix const kInitialState { 
	4, 
	1, 
	{ 
		0.0f, 0.0f, 0.0f, 0.0f 
	} 
};

/**
 * @brief Structure to store covariance of the initial state : system dimension * system dimension
 *
 */
static utility::Matrix const kInitialNoiseCov {
    4,
    4,
    { 
		1.0f, 0.0f, 0.0f, 0.0f, 
		0.0f, 1e-6f, 0.0f, 0.0f, 
		0.0f, 0.0f, 1e-6f, 0.0f, 
		0.0f, 0.0f, 0.0f, 1e-6f 
	}
};

/**
 * @brief Structure to store squared covariance of the initial state : system dimension * system dimension
 *
 */
static utility::Matrix const kSqInitialNoiseCov {
    4,
    4,
    { 
		1.0f, 0.0f, 0.0f, 0.0f, 
		0.0f, 1e-3f, 0.0f, 0.0f, 
		0.0f, 0.0f, 1e-3f, 0.0f, 
		0.0f, 0.0f, 0.0f, 1e-3f
	}
};

/**
 * @brief Structure to store measurement noise covariance : measurement dimension * measurement dimension
 *
 */
static utility::Matrix const kMeasNoiseCov { 
	2, 
	2, 
	{ 
		0.1f, 0.0f, 
		0.0f, 0.1f 
	} 
};

/**
 * @brief Structure to store inverse measurement noise covariance : measurement dimension * measurement dimension
 *
 */
static utility::Matrix const kInvMeasNoiseCov { 
	2, 
	2, 
	{ 
		10.0f, 0.0f, 
		0.0f, 10.0f
	} 
};

/**
 * @brief Structure to store process noise covariance : system dimension * system dimension
 *
 */
static utility::Matrix const kProcessNoiseCov {
    4,
    4,
    { 
		0.01f, 0.00f, 0.00f, 0.00f, 
		0.00f, 0.01f, 0.00f, 0.00f, 
		0.00f, 0.00f, 0.01f, 0.00f, 
		0.00f, 0.00f, 0.00f, 0.01f 
	}
};

/**
 * @brief 	Structure to store squared process noise covariance : system dimension * system dimension
 *
 */
static utility::Matrix const kSqProcessNoiseCov {
    4,
    4,
    { 
		0.1f, 0.0f, 0.0f, 0.0f, 
		0.0f, 0.1f, 0.0f, 0.0f, 
		0.0f, 0.0f, 0.1f, 0.0f, 
		0.0f, 0.0f, 0.0f, 0.1f 
	}
};

/**
 * @brief Identity matrix of four dimensions : system dimension * system dimension
 *
 */
static utility::Matrix const kIdentity {
    4,
    4,
    { 
		1.0f, 0.0f, 0.0f, 0.0f, 
		0.0f, 1.0f, 0.0f, 0.0f, 
		0.0f, 0.0f, 1.0f, 0.0f, 
		0.0f, 0.0f, 0.0f, 1.0f 
	}
};

} // namespace models

#endif /* MODELS_H_ */
