# Particle Filter
This GPU accelerated particle filter

## Prerequisites
The build process assumes compute capability (CC) 7.0 or greater.

An additional CC can easily be added by appending to ARCHES in the makefile.

## Performance
Come soon

## Documentation
In depth documentation can be found in the following papers:
- [Improved Parallel Resampling Methods for Particle Filtering](https://github.com/mnicely/particle_filter/blob/master/docs/Improved_Parallel_Resampling_IEEE_Nicely_041919.pdf)
- [Parallel Implementation of Resampling Methods for Particle Filtering on Graphics Processing Units](https://github.com/mnicely/particle_filter/blob/master/docs/Dissertation_Nicely_111319.pdf)
<!-- - [Marginalized Particle Filters for Mixed
Linear/Nonlinear State-space Models](http://user.it.uu.se/~thosc112/pubpdf/schongn2005.pdf) -->

## Built with
This application uses the following toolsets:
- [CUDA 9.0+](https://developer.nvidia.com/cuda-downloads)
- [CUB 1.8.0](https://nvlabs.github.io/cub/)
- [NVTX](https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/)
- C++14
- LAPACKE
- OpenBLAS
- Boost Program Options

## Tested on
This application was successfully tested on the following hardware:
- Intel i7-8700K
- NVIDIA GeForce RTX 2080
- NVIDIA GeForce GTX 1080
- NVIDIA GeForce GTX 980
- NVIDIA Titan V
- NVIDIA Titan RTX

## Usage
This application has the following options:
```bash
Program Options:
  -f [ --filter ] arg (=0)         Type of filter: Bootstrap = 0
  -p [ --particles ] arg (=65536)  Number of particles
  -s [ --samples ] arg (=500)      Number of samples to execute
  -r [ --resampling ] arg (=0)     Resampling method: Systmatic = 0, Stratified = 1, Metropolis = 2
  -m [ --mcs ] arg (=5)            Number of Monte Carlos to execute
  -c [ --create ]                  Create truth data
  -t [ --truth ]                   Use precalculate truth data
  -g [ --gpu ]                     Use GPU or CPU
  -h [ --help ]                    Display help menu.

```

An example script can be found [here]()


### Profiling with NVTX
If you want to utilizing NVTX for more in-depth profiling, you will need to build with the NVTX option.
```bash
make clean; make -j all NVTX=1
```

### Output
If everything runs successfully, you should see an output like the following:
```bash
CPU: Systematic: Monte Carlos 5: Samples 500: Particles: 65536
Data stored in ./data/estimate_CPU_Systematic_65536.txt and ./data/truth_CPU_Systematic_65536.txt
Average Times (us)
Median   Mean    StdDev  
16546    16558   128     

GPU: Systematic: Monte Carlos 5: Samples 500: Particles: 65536
Data stored in ./data/estimate_GPU_Systematic_65536.txt and ./data/truth_GPU_Systematic_65536.txt
Average Times (us)
Median   Mean    StdDev  
55       51      5
```

## TODOs
- Add performance results for TX2 and Xavier platforms
- Remove CBLAS from serial version to increase performance
- Add [Marginal Particle Filter](http://user.it.uu.se/~thosc112/pubpdf/schongn2005.pdf) implementation
