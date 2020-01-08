# Particle Filter
GPU accelerated particle filter with 4 system states and 2 measurement states. 

Models are based on research by [Thomas Schon](http://user.it.uu.se/~thosc112/index.html) and MATLAB code can be found [here](http://user.it.uu.se/~thosc112/research/rao-blackwellized-particle.html).

## Prerequisites
The build process assumes the following:
- CUDA is installed at ```/usr/local/cuda/```.
- Compute capability (CC) 7.0.

An additional CC can easily be added by appending to **ARCHES** in the makefile.

## Performance
Come soon!

## Documentation
In depth documentation can be found in the following papers:
- [Improved Parallel Resampling Methods for Particle Filtering](https://github.com/mnicely/particle_filter/blob/master/docs/Improved_Parallel_Resampling_IEEE_Nicely_041919.pdf)
- [Parallel Implementation of Resampling Methods for Particle Filtering on Graphics Processing Units](https://github.com/mnicely/particle_filter/blob/master/docs/Dissertation_Nicely_111319.pdf)
<!-- - [Marginalized Particle Filters for Mixed
Linear/Nonlinear State-space Models](http://user.it.uu.se/~thosc112/pubpdf/schongn2005.pdf) -->

## Built with
This application uses the following toolsets:
- [CUDA 9.0 or later](https://developer.nvidia.com/cuda-downloads)
- [CUB 1.7.5 or later](https://nvlabs.github.io/cub/)
- [NVTX](https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/)
- C++14
- LAPACKE
- OpenBLAS
- Boost Program Options

## Tested on
This application was successfully tested on the following:
- Intel i7-8700K
- NVIDIA GeForce RTX 2080
- NVIDIA GeForce GTX 1080
- NVIDIA GeForce GTX 980
- NVIDIA Titan V
- NVIDIA Titan RTX
- Ubuntu 16.04/18.04
- CUDA 9.X/10.X
- CUB 1.7.5/1.8.0

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
  -t [ --truth ]                   Use precalculated truth data
  -g [ --gpu ]                     Use GPU or CPU
  -h [ --help ]                    Display help menu.

```

An example script can be found [here](https://github.com/mnicely/particle_filter/blob/master/scripts/example_script.sh)


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

### Post-processing
The root mean square error of all 4 states estimates can be evaluated with the MATLAB script [analysis.m](https://github.com/mnicely/particle_filter/blob/master/data/analysis.m).

```matlab
Check GPU
    'RMSE for 65536 for Systematic'

    0.3119    0.2060    0.1792    0.1553

    'RMSE for 65536 for Stratified'

    0.3112    0.2057    0.1791    0.1553

    'RMSE for 65536 for MetropolisC2'

    0.3125    0.2069    0.1801    0.1559

    'RMSE for 1048576 for Systematic'

    0.3119    0.2061    0.1792    0.1554

    'RMSE for 1048576 for Stratified'

    0.3116    0.2060    0.1792    0.1554

    'RMSE for 1048576 for MetropolisC2'

    0.3128    0.2062    0.1793    0.1553
```

## TODOs
- Add performance results for TX2 and Xavier platforms
- Add [Marginal Particle Filter](http://user.it.uu.se/~thosc112/pubpdf/schongn2005.pdf) implementation
