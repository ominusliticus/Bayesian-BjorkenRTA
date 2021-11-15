#!/bin/bash

echo "Checking for 3rd party libraries, and installing if needed"
echo pwd
cd ../3rd_party

if [[-d fmt]]
then
    echo 'fmt library exists'
else
    echo "fmt repository doesn't exist"
    echo "cloning repository https://github.com/fmtlib/fmt.git"
    git clone --recursive https://github.com/fmtlib/fmt.git
    echo "Done"
fi

if [[-d aramadillo]]
then 
    echo "armadillo library exists"
else
    echo "armadillo library doesn't exist"
    echo "cloning repository https://gitlab.com/conradsnicta/armadillo-code.git"
    git clone --recursive https://gitlab.com/conradsnicta/armadillo-code.git
    mv armadillo-code armadillo
    echo "Done"
fi

cd ../
echo "Done installing all dependencies."

echo "Creating necessary directories"
mkdir build
mkdir output
mkdir output/aniso_hydro
mkdir output/bayes
mkdir output/CE_hydro
mdkir output/DNMR_hydro
mkdir exact
cd script/
mkdir design_points
mkdir full_outputs
mkdir hydro_simulation_points
mkdir pickle_files

cd ../
echo "Directories created"