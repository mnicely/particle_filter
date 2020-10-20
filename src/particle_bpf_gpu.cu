/**
 * @file particle_bpf_gpu.cu
 * @author Matthew Nicely (mnicely@nvidia.com)
 * @date 2020-01-06
 * @version 1.0
 * @brief Contains CUDA functions for parallel version of
 * the bootstrap particle filter
 *
 * @copyright Copyright (c) 2020
 *
 * @license This project is released under the GNU Public License
 *
 * @note * Target Processor: Intel x86
 * @n * Target Compiler: GCC 7.4.0
 * @n * NVCC Compiler: CUDA Toolkit 10.0 or later
 */

#include <cooperative_groups.h>  // cooperative groups::this_thread_block, cooperative groups::tiled_partition
#include <cub/cub.cuh>           // cub::CacheModifiedInputIterator, cub::BlockLoad, cub::BlockStore, cub::WarpReduce
#include <curand_kernel.h>       // curand_init, curand_normal, curand_uniform, curandStateXORWOW_t

#include "models.h"

namespace cg = cooperative_groups;

constexpr auto kSysDim { utility::kSysDim };    // state dimension
constexpr auto kMeasDim { utility::kMeasDim };  // measurement dimension
constexpr auto kMetropolisB { 32 };             // Iterations in Metropolis resampling
constexpr auto kWarpSize { 32 };
constexpr auto kBlocks { 20 };
constexpr auto kTAI { 256 };
constexpr auto kTME { 256 };
constexpr auto kTCE { 256 };
constexpr auto kTPT { 256 };
constexpr auto kTRI { 64 };
constexpr auto kTMR { 256 };

namespace filters {

__constant__ float c_initial_state[kSysDim] {};

__constant__ float c_meas_update[kMeasDim] {};

__constant__ float c_inv_meas_noise_cov[kMeasDim * kMeasDim] {};

__constant__ float c_process_noise_cov[kSysDim * kSysDim] {};

__constant__ float c_initial_noise_cov[kSysDim * kSysDim] {};

__device__ float d_sum_of_particle_weights {};

template<typename T>
__global__ void __launch_bounds__( kTAI )
    InitializeFilter( int const num_particles, unsigned long long int const seed, T *__restrict__ particle_state_new ) {

    auto const block { cg::this_thread_block( ) };

    typedef cub::CacheModifiedInputIterator<cub::LOAD_LDG, T>                  InputItr;
    typedef cub::BlockLoad<T, kTAI, kSysDim, cub::BLOCK_LOAD_WARP_TRANSPOSE>   BlockLoad;
    typedef cub::BlockStore<T, kTAI, kSysDim, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStore;

    __shared__ union TempStorage {
        typename BlockLoad::TempStorage  load;
        typename BlockStore::TempStorage store;
    } temp_storage;

    unsigned int loop = blockIdx.x * blockDim.x * kSysDim;

    for ( unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_particles;
          tid += blockDim.x * gridDim.x ) {

        T thread_data[kSysDim] {};
        T random_nums[kSysDim] {};

        curandState local_state {};

        curand_init( static_cast<unsigned long long int>( seed + tid ), 0, 0, &local_state );

#pragma unroll kSysDim
        for ( T &x : random_nums ) {
            x = curand_normal( &local_state );
        }

#pragma unroll kSysDim
        for ( int i = 0; i < kSysDim; i++ ) {
            thread_data[i] = c_initial_state[i];
#pragma unroll kSysDim
            for ( int j = 0; j < kSysDim; j++ ) {
                thread_data[i] += c_initial_noise_cov[i * kSysDim + j] * random_nums[j];
            }
        }

        BlockStore( temp_storage.store ).Store( particle_state_new + loop, thread_data );

        block.sync( );

        // grid size * number of system states
        loop += blockDim.x * gridDim.x * kSysDim;
    }
}

template<typename T>
__global__ void __launch_bounds__( kTME ) ComputeMeasErrors( int const num_particles,
                                                             T const *__restrict__ particle_state_new,
                                                             T *__restrict__ particle_weights,
                                                             T *__restrict__ particle_state ) {

    auto const block { cg::this_thread_block( ) };

    /*
     * Sum particle weights using BlockReduce in this kernel
     * to save on global memory loads later
     * Note that d_sum_of_particle_weights is reset to zero in
     * void ComputeParticleTransitionCuda
     */

    typedef cub::CacheModifiedInputIterator<cub::LOAD_LDG, T>                  InputItr;
    typedef cub::BlockLoad<T, kTME, kSysDim, cub::BLOCK_LOAD_WARP_TRANSPOSE>   BlockLoad;
    typedef cub::BlockStore<T, kTME, kSysDim, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStore;
    typedef cub::BlockReduce<T, kTME>                                          BlockReduce;

    __shared__ union TempStorage {
        typename BlockLoad::TempStorage   load;
        typename BlockStore::TempStorage  store;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    unsigned int loop { blockIdx.x * blockDim.x * kSysDim };

    for ( unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_particles;
          tid += blockDim.x * gridDim.x ) {

        T thread_data[kSysDim] {};
        T estimates[kMeasDim] {};
        T errors[kMeasDim] {};

        BlockLoad( temp_storage.load ).Load( InputItr( particle_state_new + loop ), thread_data );

        block.sync( );

        models::MeasModelMath( thread_data, estimates );

        T sum {};

#pragma unroll kMeasDim
        for ( int i = 0; i < kMeasDim; i++ ) {
            errors[i] = c_meas_update[i] - estimates[i];
            errors[i] *= errors[i];
        }

#pragma unroll kMeasDim
        for ( int i = 0; i < kMeasDim; i++ ) {
#pragma unroll kMeasDim
            for ( int j = 0; j < kMeasDim; j++ ) {
                sum += c_inv_meas_noise_cov[i * kMeasDim + j] * errors[j];
            }
        }

        float particle_weight { expf( sum * -0.5f ) };

        particle_weights[tid] = particle_weight;

        float blockSum { BlockReduce( temp_storage.reduce ).Sum( particle_weight ) };

        block.sync( );

        if ( threadIdx.x == 0 ) {
            atomicAdd( &d_sum_of_particle_weights, blockSum );
        }

        BlockStore( temp_storage.store ).Store( particle_state + loop, thread_data );

        block.sync( );

        // grid size * number of system states
        loop += blockDim.x * gridDim.x * kSysDim;
    }
}

template<typename T>
__global__ void __launch_bounds__( kTCE ) ComputeEstimates( int const num_particles,
                                                            int const time_step,
                                                            int const resampling_method,
                                                            T const *__restrict__ particle_state,
                                                            T *__restrict__ filtered_estimates,
                                                            T *__restrict__ particle_weights ) {

    auto const block   = cg::this_thread_block( );
    auto const tile_32 = cg::tiled_partition( block, 32 );
    auto const laneID  = tile_32.thread_rank( );

    typedef cub::CacheModifiedInputIterator<cub::LOAD_LDG, T>                InputItr;
    typedef cub::BlockLoad<T, kTCE, kSysDim, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;
    typedef cub::WarpReduce<T>                                               WarpReduce;

    __shared__ union TempStorage {
        typename BlockLoad::TempStorage  load;
        typename WarpReduce::TempStorage warpReduce[kWarpSize];
    } temp_storage;

    __shared__ T s_partial_reduce[kWarpSize];  // kWarpSize is 32. Allows for 32
                                               // warps (1024 threads)
    __shared__ T s_final_reduce[kSysDim];

    unsigned int const warp_id { threadIdx.x >> 5 };

    unsigned int loop { blockIdx.x * blockDim.x * kSysDim };

    for ( unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_particles;
          tid += blockDim.x * gridDim.x ) {

        if ( warp_id == 0 ) {
            s_partial_reduce[laneID] = 0;  // Initialize shared memory
        }

        T thread_data[kSysDim] {};
        T val {};
        T normalized { particle_weights[tid] / d_sum_of_particle_weights };

        // Load a segment of consecutive items that are blocked across threads
        BlockLoad( temp_storage.load ).Load( InputItr( particle_state + loop ), thread_data );

        block.sync( );

#pragma unroll kSysDim
        for ( int i = 0; i < kSysDim; i++ ) {

            thread_data[i] *= normalized;

            // Each warp perform reduction
            val = WarpReduce( temp_storage.warpReduce[warp_id] ).Sum( thread_data[i] );

            // Write reduced value to shared memory
            if ( laneID == 0 ) {
                s_partial_reduce[warp_id] = val;
            }

            block.sync( );  // Wait for all partial reductions

            // Read from shared memory only if that warp existed
            if ( warp_id == 0 ) {
                val = WarpReduce( temp_storage.warpReduce[0] ).Sum( s_partial_reduce[laneID] );
            }

            if ( threadIdx.x == 0 ) {
                s_final_reduce[i] = val;
            }

            block.sync( );  // Wait for final reduction
        }

        /*
         * For systematic and stratified resampling, normalized weights are
         * need. To save on global loads in future kernels, particle_weights is
         * normalized and written back to globals.
         * For Metropolis, we normalize for filter estimates but don't store
         * normalized weights back to global.
         */
        if ( resampling_method != static_cast<int>( utility::Method::kMetropolisC2 ) ) {
            particle_weights[tid] = normalized;
        }

        if ( threadIdx.x < kSysDim ) {
            atomicAdd( &filtered_estimates[time_step * kSysDim + laneID], s_final_reduce[laneID] );
        }

        // grid size * number of system states
        loop += blockDim.x * gridDim.x * kSysDim;
    }
}

template<typename T>
__device__ void ResamplingUpPerWarp( cg::thread_block_tile<kWarpSize> const &tile_32,
                                     unsigned int const &                    tid,
                                     int const &                             num_particles,
                                     T const &                               distro,
                                     T *                                     shared,
                                     T *__restrict__ prefix_sum,
                                     int *__restrict__ resampling_index_up ) {

    T const    tidf { static_cast<T>( tid ) };
    auto const t { tile_32.thread_rank( ) };

    int l {};
    int idx {};
    T   a {};
    T   b {};

    bool mask { true };

    if ( tid < num_particles - kWarpSize - l ) {
        shared[t]             = prefix_sum[tid + l];
        shared[t + kWarpSize] = prefix_sum[tid + kWarpSize + l];
    }

    // Distribution will be the same for each Monte Carlo
    T const draw = ( distro + tidf ) / num_particles;

    tile_32.sync( );

    while ( tile_32.any( mask ) ) {
        if ( tid < num_particles - ( kTRI )-l ) {

            a = prefix_sum[tid + kWarpSize + l];
            b = prefix_sum[tid + kTRI + l];

#pragma unroll kWarpSize
            for ( int i = 0; i < kWarpSize; i++ ) {
                mask = shared[t + i] < draw;
                if ( mask ) {
                    idx++;
                }
            }
            l += kWarpSize;
            shared[t]             = a;
            shared[t + kWarpSize] = b;

            tile_32.sync( );
        } else {
            while ( mask && tid < ( num_particles - l ) ) {
                mask = prefix_sum[tid + l] < draw;
                if ( mask ) {
                    idx++;
                }
                l++;
            }
        }

        tile_32.sync( );
    }
    resampling_index_up[tid] = idx;
}

template<typename T>
__global__ void __launch_bounds__( kTRI )
    ComputeResampleIndexSysUpSharedPrefetch64( int const                    num_particles,
                                               unsigned long long int const seed,
                                               int const                    resampling_method,
                                               int *__restrict__ resampling_index_up,
                                               T *__restrict__ prefix_sum ) {

    auto const tile_32 = cg::tiled_partition<kWarpSize>( cg::this_thread_block( ) );

    __shared__ T s_warp_0[kTRI];
    __shared__ T s_warp_1[kTRI];

    // Setting prefix_sum[n - 1] in each block versus call a separate kernel
    // beforehand. Set last value in prefix-sum to 1.0f
    if ( threadIdx.x == 0 ) {
        prefix_sum[num_particles - 1] = 1.0f;  //
    }

    for ( unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_particles;
          tid += blockDim.x * gridDim.x ) {

        curandStateXORWOW_t local_state {};

        T distro {};

        if ( resampling_method == static_cast<int>( utility::Method::kSystematic ) ) {
            curand_init( seed, 0, 0, &local_state );
            distro = curand_uniform( &local_state );
        } else if ( resampling_method == static_cast<int>( utility::Method::kStratified ) ) {
            curand_init( seed + tid, 0, 0, &local_state );
            distro = curand_uniform( &local_state );
        }

        if ( threadIdx.x < kWarpSize ) {
            ResamplingUpPerWarp( tile_32, tid, num_particles, distro, s_warp_0, prefix_sum, resampling_index_up );
        } else {
            ResamplingUpPerWarp( tile_32, tid, num_particles, distro, s_warp_1, prefix_sum, resampling_index_up );
        }
    }
}

template<typename T>
__device__ void ResamplingDownPerWarp( cg::thread_block_tile<kWarpSize> const &tile_32,
                                       unsigned int const &                    tid,
                                       int const &                             num_particles,
                                       T const &                               distro,
                                       T *                                     shared,
                                       T *__restrict__ prefix_sum,
                                       int *__restrict__ resampling_index_down ) {

    T const    tidf { static_cast<T>( tid ) };
    auto const t { tile_32.thread_rank( ) };

    int l {};
    int idx {};
    T   a {};
    T   b {};

    bool mask { false };

    // Preload in into shared memory
    if ( tid >= kWarpSize + l ) {
        shared[t]             = prefix_sum[tid - kWarpSize - l];
        shared[t + kWarpSize] = prefix_sum[tid - l];
    }

    // Distribution will be the same for each Monte Carlo
    T const draw { ( distro + tidf ) / num_particles };

    tile_32.sync( );

    while ( !tile_32.all( mask ) ) {

        if ( tid >= kTRI + l ) {
            a = prefix_sum[tid - ( kTRI )-l];
            b = prefix_sum[tid - kWarpSize - l];

#pragma unroll
            for ( int i = 1; i < kWarpSize + 1; i++ ) {
                mask = shared[t + kWarpSize - i] < draw;
                if ( !mask ) {
                    idx--;
                }
            }
            l += kWarpSize;
            shared[t]             = a;
            shared[t + kWarpSize] = b;
            tile_32.sync( );

        } else {

            while ( !mask ) {
                if ( tid > l ) {
                    mask = prefix_sum[tid - ( l + 1 )] < draw;
                } else {
                    mask = true;
                }
                if ( !mask ) {
                    idx--;
                }
                l++;
            }
        }

        tile_32.sync( );
    }
    resampling_index_down[tid] = idx;
}

template<typename T>
__global__ void __launch_bounds__( kTRI )
    ComputeResampleIndexSysDownSharedPrefetch64( int const                    num_particles,
                                                 unsigned long long int const seed,
                                                 int const                    resampling_method,
                                                 int *__restrict__ resampling_index_down,
                                                 T *__restrict__ prefix_sum ) {

    auto const tile_32 = cg::tiled_partition<kWarpSize>( cg::this_thread_block( ) );

    __shared__ T s_warp_0[kTRI];
    __shared__ T s_warp_1[kTRI];

    // Setting prefix_sum_particle_weights[n - 1] in each block versus call a
    // separate kernel beforehand
    if ( threadIdx.x == 0 ) {
        prefix_sum[num_particles - 1] = 1.0f;
    }

    for ( unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_particles;
          tid += blockDim.x * gridDim.x ) {

        curandStateXORWOW_t local_state {};

        T distro {};

        if ( resampling_method == static_cast<int>( utility::Method::kSystematic ) ) {
            curand_init( seed, 0, 0, &local_state );
            distro = curand_uniform( &local_state );
        } else if ( resampling_method == static_cast<int>( utility::Method::kStratified ) ) {
            curand_init( seed + tid, 0, 0, &local_state );
            distro = curand_uniform( &local_state );
        }

        if ( threadIdx.x < kWarpSize ) {
            ResamplingDownPerWarp( tile_32, tid, num_particles, distro, s_warp_0, prefix_sum, resampling_index_down );
        } else {
            ResamplingDownPerWarp( tile_32, tid, num_particles, distro, s_warp_1, prefix_sum, resampling_index_down );
        }
    }
}

template<typename T>
__global__ void __launch_bounds__( kTMR ) ComputeResampleIndexMetropolisC2( int const                    num_particles,
                                                                            unsigned long long int const seed,
                                                                            T const *__restrict__ particle_weights,
                                                                            int *__restrict__ resampling_index_down ) {

    for ( unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_particles;
          tid += blockDim.x * gridDim.x ) {

        unsigned int idx { tid };
        unsigned int key {};
        unsigned int warp {};

        T den { particle_weights[tid] };
        T num {};
        T random_num {};

        curandStateXORWOW_t    local_state {};
        curandStateXORWOW_t    rand_state {};
        unsigned long long int local_seed { static_cast<unsigned long long int>( tid >> 5 ) + seed };

        // same random number for 0-based warp entire grid
        curand_init( local_seed, 0, 0, &local_state );
        curand_init( local_seed, 0, 0, &rand_state );

        // Calculate s(warp) using warp index. Threads in warp have same value
        int ss { kWarpSize };           // Size of segment
        int sc { num_particles / ss };  // The number of segments
        int dc { ss };

        for ( int i = 0; i < kMetropolisB; i++ ) {

            warp = static_cast<unsigned int>( curand_uniform( &local_state ) *
                                              ( sc - 1 ) );  // Random number [0 -> number of warps]

            random_num = curand_uniform( &rand_state );
            key        = static_cast<unsigned int>( curand_uniform( &rand_state ) * ( dc - 1 ) );
            key        = warp * dc + key;
            num        = particle_weights[key];

            if ( random_num <= ( num / den ) ) {
                den = num;
                idx = key;
            }
        }
        resampling_index_down[tid] = idx;
    }
}

template<typename T>
__global__ void __launch_bounds__( kTPT ) ComputeParticleTransition( int const                    num_particles,
                                                                     unsigned long long int const seed,
                                                                     int const                    resampling_method,
                                                                     int const *__restrict__ resampling_index_up,
                                                                     int const *__restrict__ resampling_index_down,
                                                                     T const *__restrict__ particle_state,
                                                                     T *__restrict__ particle_state_new ) {

    auto const block { cg::this_thread_block( ) };

    typedef cub::BlockStore<T, kTPT, kSysDim, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStore;

    __shared__ typename BlockStore::TempStorage temp_storage;

    unsigned int loop { blockIdx.x * blockDim.x * kSysDim };

    for ( unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_particles;
          tid += blockDim.x * gridDim.x ) {

        int idx {};
        if ( resampling_method != static_cast<int>( utility::Method::kMetropolisC2 ) ) {
            idx = static_cast<int>( tid ) + resampling_index_up[tid] + resampling_index_down[tid];
        } else {
            idx = resampling_index_down[tid];
        }

        T model_update[kSysDim] {};
        T thread_data[kSysDim] {};
        T random_nums[kSysDim] {};

        curandState local_state {};

        curand_init( static_cast<unsigned long long int>( seed + tid ), 0, 0, &local_state );

#pragma unroll kSysDim
        for ( int i = 0; i < kSysDim; i++ ) {
            thread_data[i] = particle_state[idx * kSysDim + i];
            random_nums[i] = curand_normal( &local_state );
        }

        models::SysModelMath( thread_data, model_update );

        // Reuse thread_data to ease register pressure
#pragma unroll kSysDim
        for ( int i = 0; i < kSysDim; i++ ) {
            thread_data[i] = model_update[i];
#pragma unroll kSysDim
            for ( int j = 0; j < kSysDim; j++ ) {
                thread_data[i] += c_process_noise_cov[i * kSysDim + j] * random_nums[j];
            }
        }

        BlockStore( temp_storage ).Store( particle_state_new + loop, thread_data );

        block.sync( );

        // grid size * number of system states
        loop += blockDim.x * gridDim.x * kSysDim;
    }
}

// Wrappers
template<typename T>
void InitializeFilterCuda( int const &         sm_count,
                           cudaStream_t const *streams,
                           int const &         num_particles,
                           T const *           pin_sq_initial_noise_cov,
                           cudaEvent_t *       events,
                           T *                 particle_state_new ) {

    int const threads_per_block { kTAI };
    int const blocks_per_grid { kBlocks * sm_count };

    unsigned long long int seed { static_cast<unsigned long long int>( clock( ) ) };

    CUDA_RT_CALL( cudaMemcpyToSymbolAsync( c_initial_noise_cov,
                                           pin_sq_initial_noise_cov,
                                           kSysDim * kSysDim * sizeof( T ),
                                           0,
                                           cudaMemcpyHostToDevice,
                                           streams[0] ) );

    void *args[] { const_cast<int *>( &num_particles ), &seed, &particle_state_new };

    CUDA_RT_CALL( cudaLaunchKernel(
        reinterpret_cast<void *>( &InitializeFilter<T> ), blocks_per_grid, threads_per_block, args, 0, streams[0] ) );
}

template<typename T>
void ComputeMeasErrorsCuda( int const &         sm_count,
                            cudaStream_t const *streams,
                            int const &         num_particles,
                            T const *           pin_inv_meas_noise_cov,
                            T const *           pin_meas_update,
                            T const *           particle_state_new,
                            cudaEvent_t *       events,
                            T *                 particle_weights,
                            T *                 particle_state ) {

    int const threads_per_block { kTME };
    int const blocks_per_grid { kBlocks * sm_count };

    CUDA_RT_CALL( cudaMemcpyToSymbolAsync( c_inv_meas_noise_cov,
                                           pin_inv_meas_noise_cov,
                                           kMeasDim * kMeasDim * sizeof( T ),
                                           0,
                                           cudaMemcpyHostToDevice,
                                           streams[1] ) );
    CUDA_RT_CALL( cudaEventRecord( events[1], streams[1] ) );

    CUDA_RT_CALL( cudaMemcpyToSymbolAsync(
        c_meas_update, pin_meas_update, kMeasDim * sizeof( T ), 0, cudaMemcpyHostToDevice, streams[0] ) );

    // Wait for cudaMemcpyToSymbolAsync -> c_inv_meas_noise_cov
    CUDA_RT_CALL( cudaStreamWaitEvent( streams[0], events[1], 0 ) );

    void *args[] { const_cast<int *>( &num_particles ), &particle_state_new, &particle_weights, &particle_state };

    CUDA_RT_CALL( cudaLaunchKernel(
        reinterpret_cast<void *>( &ComputeMeasErrors<T> ), blocks_per_grid, threads_per_block, args, 0, streams[0] ) );
}

template<typename T>
void ComputeEstimatesCuda( int const &         sm_count,
                           cudaStream_t const *streams,
                           int const &         num_particles,
                           int const &         time_step,
                           int const &         resampling_method,
                           T const *           particle_state,
                           cudaEvent_t *       events,
                           T *                 filtered_estimates,
                           T *                 particle_weights ) {

    int const threads_per_block { kTCE };
    int const blocks_per_grid { kBlocks * sm_count };

    void *args[] { const_cast<int *>( &num_particles ),
                   const_cast<int *>( &time_step ),
                   const_cast<int *>( &resampling_method ),
                   &particle_state,
                   &filtered_estimates,
                   &particle_weights };

    CUDA_RT_CALL( cudaLaunchKernel(
        reinterpret_cast<void *>( &ComputeEstimates<T> ), blocks_per_grid, threads_per_block, args, 0, streams[0] ) );
}

template<typename T>
void ComputeResampleIndexCuda( int const &         sm_count,
                               cudaStream_t const *streams,
                               int const &         num_particles,
                               int const &         time_step,
                               int const &         resampling_method,
                               T const *           particle_weights,
                               T *                 prefix_sum_particle_weights,
                               cudaEvent_t *       events,
                               int *               resampling_index_up,
                               int *               resampling_index_down ) {

    unsigned long long int seed { static_cast<unsigned long long int>( clock( ) ) };

    // If Systematic and Stratified
    if ( resampling_method != static_cast<int>( utility::Method::kMetropolisC2 ) ) {

        int const threads_per_block { kTRI };

        int blocks_per_grid {};
        if ( num_particles > 100000 ) {
            blocks_per_grid = 2 * kBlocks * sm_count;
        }  // Better performance with more blocks
        else {
            blocks_per_grid = kBlocks * sm_count;
        }  // Better performance with fewer blocks

        //*********************** Perform Cumulative Sum
        //***************************
        void * d_temp_storage { nullptr };
        size_t temp_storage_bytes {};

        // Determine temporary device storage requirements for inclusive prefix
        // sum on normalized particleWeights
        cub::DeviceScan::InclusiveSum( d_temp_storage,
                                       temp_storage_bytes,
                                       particle_weights,
                                       prefix_sum_particle_weights,
                                       num_particles,
                                       streams[0],
                                       false );

        // Allocate temporary storage
        CUDA_RT_CALL( cudaMalloc( &d_temp_storage, temp_storage_bytes ) );

        // Run inclusive prefix sum
        cub::DeviceScan::InclusiveSum( d_temp_storage,
                                       temp_storage_bytes,
                                       particle_weights,
                                       prefix_sum_particle_weights,
                                       num_particles,
                                       streams[0],
                                       false );

        // Sync cumulative sum
        CUDA_RT_CALL( cudaEventRecord( events[1], streams[0] ) );

        void *args_up[] { const_cast<int *>( &num_particles ),
                          &seed,
                          const_cast<int *>( &resampling_method ),
                          &resampling_index_up,
                          &prefix_sum_particle_weights };

        CUDA_RT_CALL( cudaLaunchKernel( reinterpret_cast<void *>( &ComputeResampleIndexSysUpSharedPrefetch64<T> ),
                                        blocks_per_grid,
                                        threads_per_block,
                                        args_up,
                                        0,
                                        streams[0] ) );

        CUDA_RT_CALL( cudaStreamWaitEvent( streams[1], events[1], 0 ) );  // Wait for InclusiveSum

        void *args_down[] { const_cast<int *>( &num_particles ),
                            &seed,
                            const_cast<int *>( &resampling_method ),
                            &resampling_index_down,
                            &prefix_sum_particle_weights };

        CUDA_RT_CALL( cudaLaunchKernel( reinterpret_cast<void *>( &ComputeResampleIndexSysDownSharedPrefetch64<T> ),
                                        blocks_per_grid,
                                        threads_per_block,
                                        args_down,
                                        0,
                                        streams[1] ) );

        CUDA_RT_CALL( cudaEventRecord( events[0], streams[1] ) );

    } else {

        int const threads_per_block { kTMR };
        int const blocks_per_grid { kBlocks * sm_count };

        void *args[] { const_cast<int *>( &num_particles ), &seed, &particle_weights, &resampling_index_down };

        CUDA_RT_CALL( cudaLaunchKernel( reinterpret_cast<void *>( &ComputeResampleIndexMetropolisC2<T> ),
                                        blocks_per_grid,
                                        threads_per_block,
                                        args,
                                        0,
                                        streams[0] ) );

        CUDA_RT_CALL( cudaEventRecord( events[0], streams[0] ) );
    }
}

template<typename T>
void ComputeParticleTransitionCuda( int const &         sm_count,
                                    cudaStream_t const *streams,
                                    int const &         num_particles,
                                    int const &         resampling_method,
                                    T const *           pin_sq_process_noise_cov,
                                    T const *           particle_state,
                                    int const *         resampling_index_up,
                                    int const *         resampling_index_down,
                                    cudaEvent_t *       events,
                                    T *                 particle_state_new ) {

    // Get d_sum_of_particle_weights address for reset
    float *h_sum_of_particle_weights;
    CUDA_RT_CALL( cudaGetSymbolAddress( ( void ** )&h_sum_of_particle_weights, d_sum_of_particle_weights ) );

    unsigned long long int seed { static_cast<unsigned long long int>( clock( ) ) };

    int const threads_per_block { kTPT };
    int const blocks_per_grid { kBlocks * sm_count };

    CUDA_RT_CALL( cudaMemcpyToSymbolAsync( c_process_noise_cov,
                                           pin_sq_process_noise_cov,
                                           kSysDim * kSysDim * sizeof( T ),
                                           0,
                                           cudaMemcpyHostToDevice,
                                           streams[0] ) );

    void *args[] { const_cast<int *>( &num_particles ),
                   &seed,
                   const_cast<int *>( &resampling_method ),
                   &resampling_index_up,
                   &resampling_index_down,
                   &particle_state,
                   &particle_state_new };

    // Systematic and Stratified must wait on
    // ComputeResampleIndexSysDownSharedPrefetch64
    if ( resampling_method != static_cast<int>( utility::Method::kMetropolisC2 ) ) {
        CUDA_RT_CALL( cudaStreamWaitEvent( streams[0], events[0], 0 ) );
    }  // Wait for ComputeResampleIndexSysDownSharedPrefetch64
    else {
        CUDA_RT_CALL( cudaStreamWaitEvent( streams[1], events[0], 0 ) );
    }  // Wait for ComputeResampleIndexMetropolisC2

    // Reset d_sum_of_particle_weights before next time step
    // If Metropolis, make sure it's not reset before ComputeEstimates is
    // finished
    CUDA_RT_CALL( cudaMemsetAsync( h_sum_of_particle_weights, 0, sizeof( T ), streams[1] ) );

    CUDA_RT_CALL( cudaLaunchKernel( reinterpret_cast<void *>( &ComputeParticleTransition<T> ),
                                    blocks_per_grid,
                                    threads_per_block,
                                    args,
                                    0,
                                    streams[0] ) );
}

// Explicit specializations needed to generate code
template void InitializeFilterCuda<float>( int const &         sm_count,
                                           cudaStream_t const *streams,
                                           int const &         num_particles,
                                           float const *       pin_sq_initial_noise_cov,
                                           cudaEvent_t *       events,
                                           float *             particle_state_new );

template void ComputeMeasErrorsCuda<float>( int const &         sm_count,
                                            cudaStream_t const *streams,
                                            int const &         num_particles,
                                            float const *       pin_inv_meas_noise_cov,
                                            float const *       pin_meas_update,
                                            float const *       particle_state_new,
                                            cudaEvent_t *       events,
                                            float *             particle_weights,
                                            float *             particle_state );

template void ComputeEstimatesCuda<float>( int const &         sm_count,
                                           cudaStream_t const *streams,
                                           int const &         num_particles,
                                           int const &         time_step,
                                           int const &         resampling_method,
                                           float const *       particle_state,
                                           cudaEvent_t *       events,
                                           float *             filtered_estimates,
                                           float *             particle_weights );

template void ComputeResampleIndexCuda<float>( int const &         sm_count,
                                               cudaStream_t const *streams,
                                               int const &         num_particles,
                                               int const &         time_step,
                                               int const &         resampling_method,
                                               float const *       particle_weights,
                                               float *             prefix_sum_particle_weights,
                                               cudaEvent_t *       events,
                                               int *               resampling_index_up,
                                               int *               resampling_index_down );

template void ComputeParticleTransitionCuda<float>( int const &         sm_count,
                                                    cudaStream_t const *streams,
                                                    int const &         num_particles,
                                                    int const &         resampling_method,
                                                    float const *       pin_sq_process_noise_cov,
                                                    float const *       particle_state,
                                                    int const *         resampling_index_up,
                                                    int const *         resampling_index_down,
                                                    cudaEvent_t *       events,
                                                    float *             particle_state_new );
} /* namespace filters */
