# About

This project explores concepts in *Bayesian Inference* by applying them to the 
toy model of Bjorken expansion.
The topics we study include  
&#9745; Parameter Estimation  
&#9744; Model Selection  
&#9744; Model Mixing  

## Repository Breakdown

There are two components to this repository.
The first is the C++ component, which implements the hdyrodynamic code for fast
simulation.
The second component is the Bayesian inferencing technology implemented in 
python.

# Getting Started

## C++ setup
Before running the C++ code, make sure all the dependencies are installed.
To get the aramadillo dependency, run the following command  
![#f03c15](https://via.placeholder.com/15/f03c15/f03c15.png)
this script creates directories
![#f03c15](https://via.placeholder.com/15/f03c15/f03c15.png)
```terminal
sh setup.sh
```
Armadillo also needs link to the open OpenBLAS library, so you will have to 
install `libopenblas`.  
The C++ code runs in parallel using the OpenMP library.
This means you have to install `libomp`.  
You can change whether the C++ code runs in parallel or single-core by
adjusting the `USE_PARALLEL` flag in `include/config.hpp`.  
The system requirements to run the C++ code are
- at least 4 cores
- compiler support for C++17 or higher  

### Debugging Failed Setup
To test that the C++ code runs successfully, run the following command  
![#f03c15](https://via.placeholder.com/15/f03c15/f03c15.png)
this script creates directories
![#f03c15](https://via.placeholder.com/15/f03c15/f03c15.png)
```terminal
sh test_run.sh
```
If this test fails, it probably means you don't have all the dependencies
installed.
If you are running on linux you can type `ldd build/exact_solution.x` to 
see what shared libraries the executable links with.
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
The equivalent command for `ldd` on MacOS is `otool -L`.

## Python Setup
The python version should be higher than 3.0 (preferably >3.8).
The libraries you need to run the python code are
(these libraries are fairly version stable)
|                   |                 |
| :---              | :---            |
| `matplotlib`      | `pydoe`         | 
| `multiprocessing` | `scipy`         |
| `numpy`           | `seaborn`       |
| `panda`           | `scikit-learn`  |
| `pickle`          | `subprocess`    |
| `ptemcee`         | `tqdm`          |

Note that on the default MacOS python distribution, the pickling of 
inner functions is not defined.
This means the `multiprocessing` cannot parallelize the code executions.
If you are running a non-default python version on your Mac and you want
the parallelization, please edit the `HydroCodeAPI.py` file accordingly.  
![#f03c15](https://via.placeholder.com/15/f03c15/f03c15.png)
Please note that the python code does create directories.
![#f03c15](https://via.placeholder.com/15/f03c15/f03c15.png)
   

# Workflow

The most important data structure for the python code is the 
parameters dictionary.
This get processed and fed to the C++ code command line interface.
It is simple to create on instance of it and pass around as necessary.
Your default parameter dictionary should look something like this
```python
local_params = {
    'tau_0': 0.1,               # start time for simulation
    'tau_f': 12.1,              # stop time for simulation
    'Lambda_0': 0.5 / 0.197,    # to be replaced with initial temperature
    'xi_0': -0.90,              # to be replaced with initial shear stress
    'alpha_0': 0.655,           # to be replaced with initial bulk stress
    'mass': 0.2 / 0.197,        # mass of particles in QGP
    'C': 5 / (4 * pi),          # relaxation time constant 
    'hydro_type': 0             # which hydro model to run
}
```
The `hydro_type`'s are as follows
| Name           | code |
| :---           | :--: |
| Chapman-Enskog | 0 |
| DNMR           | 1 |
| VAH            | 2 |
| Modified VAH   | 3 |
| Exact RTA Soln | 4 |  
The `hydro_type` field is automatically modified by `HydroCodeAPI`, in
accordance with which hydro name you give it.

### Example Workflow
Assuming that you have already run the `make` command in the top directory of this
project, an example parameter estimation workflow would look something like this
```python
from numpy import array, pi, linspace
import HydroCodeAPI as HCA
import HydroEmulation as HE
import HydroBayesianAnalysis as HBA
# some setup
output_path = './'
parameter_names = ['C']
parameter_ranges = array([[1 / (4 * pi), [10 / (4 * pi)]])
simulation_taus = linspace(5.1, 12.1, 8, endpoint=True)
# instantiate HydroCodeAPI
code_api = HCA(str(Path(f'{output_path}/swap').absolute()))
# instantiate HydroEmulation
emulator_class = HE(
    hca=code_api,
    params_dict=local_params,   # Have not defined here see above
    parameter_names=parameter_names,
    parameter_ranges=parameter_ranges,
    simulation_taus=simulation_taus,
    hydro_names=code_api.hydro_names,
    use_existing_emulators=False,
    use_PT_PL=True,
    output_dir=output_dir,
    samples_per_feature=points_per_feat)
# Instantiate HydroBayesianAnalysis
ba_class = HBA(
    default_params=local_params,
    parameter_names=parameter_names,
    parameter_ranges=parameter_ranges,
    simulation_taus=simulation_taus)
# Run MCMC sampler
ba_class.RunMCMC(
    nsteps=number_steps,
    nburn=50,
    ntemps=20,
    exact_observables=exact_pseudo,
    exact_error=pseudo_error,
    GP_emulators=emulator_class.GP_emulators,
    read_from_file=False)
```

# How to cite
The paper is currently still being published, but this document will be updated with the relevant sources when everything is read.
