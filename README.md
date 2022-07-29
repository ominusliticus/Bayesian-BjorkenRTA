# About

This project explores concepts in *Bayesian Inference* by applying them to the 
toy model of Bjorken expansion.
The topics we study include
- [x] Parameter Estimation
- [] Model Selection
- [] Model Mixing

## Repository Breakdown

There are two components to this repository.
The first is the C++ component, which implents the hdyrodynamic code for fast
simulation.
The second component is the Bayesian inferencing technology implemented in 
python.

# Getting Started

## C++ setup
Before running the C++ code, make sure all the dependencies are install.
To get the aramadillo dependency, run the following command
(<span style="color:red">this script creates directories</span>)
```terminal
sh setup.sh
```
Armadillo also need link to the open OpenBLAS library, so you will have to 
install `libopenblas`.
The C++ code runs in parallel using the OpenMP library.
This means you have to install `libomp`.
You can change whether the C++ code runs in parallel or single-core by
adjusting the flag `USE_PARALLEL` flag `include/config.hpp`.  
The system requirements to run the C++ code are
- at least 4 cores
- compiler support for C++17 or higher  

### Debugging Failed Setup
To test that the C++ code runs successfully, run the following command
(<span style="color:red">this script creates directories</span>)
```terminal
sh test_run.sh
```
If this test fails, it probably means you don't have all the dependencies
install.
If you are running on linux you can type `ldd build/exact_solution.x` to 
what libraries the executable links to.
The ouput should look something like this
```terminal
$ ldd build/exact_solution.x 
	linux-vdso.so.1 (0x00007fff03188000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f5ed1c71000)
	libopenblas.so.0 => /lib/x86_64-linux-gnu/libopenblas.so.0 (0x00007f5ecfae4000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f5ecf8ca000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f5ecf77b000)
	libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007f5ecf736000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f5ecf71b000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f5ecf527000)
    /lib64/ld-linux-x86-64.so.2 (0x00007f5ed1d13000)
	libgfortran.so.5 => /lib/x86_64-linux-gnu/libgfortran.so.5 (0x00007f5ecf26d000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f5ecf267000)
	libquadmath.so.0 => /lib/x86_64-linux-gnu/libquadmath.so.0 (0x00007f5ecf21d000)
```
The equivalent command for for `ldd` on MacOS is `otool -L`.

## Python Setup
The python version should be higher than 3.0 (preferably 3.8>).
The libraries you need to run the python code are
(these libraries are fairly version stabl)
<div style="display: grid; grid-template-columns: 1fr 1fr;">
  <div style="text-align: just;" markdown="1">
- `matplotlib`
- `multiprocessing`
- `numpy`
- `panda`
- `pickle`
- `ptemcee`
  </div>
  <div style="text-algin: just;" markdown="1">
- `pydoe`
- `scipy`
- `seaborn`
- `scikit-learn`
- `subprocess`
- `tqdm`
  </div>
</div>


# How to cite
The paper is currently still being published, but this document will be updated with the relevant sources when everything is read.
