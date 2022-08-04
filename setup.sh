#!/bin/bash

echo "Checking for 3rd party libraries, and installing if needed"
echo pwd
mkdir -p 3rd_party
cd 3rd_party

if [[ ! -d aramadillo ]]
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

