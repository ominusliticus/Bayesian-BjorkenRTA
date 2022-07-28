#!/bin/bash

echo "Checking for 3rd party libraries, and installing if needed"
echo pwd
cd ../3rd_party

if [[ -d aramadillo ]]
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
mkdir -p build
mkdir -p output
mkdir -p output/exact
mkdir -p output/aniso_hydro
mkdir -p output/bayes
mkdir -p output/CE_hydro
mkdir -p output/DNMR_hydro
mkdir -p scripts/design_points
mkdir -p scripts/full_outputs
mkdir -p scripts/hydro_simulation_points
mkdir -p scripts/pickle_files
mkdir -p scripts/swap

echo "Directories created"
