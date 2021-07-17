#!/bin/bash

echo "Checking for 3rd party libraries, and installing if needed"
echo pwd
if [[-d 3rd_party]]
then
    cd 3rd_party
else
    cd ../3rd_party
fi

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
    echo "Done"
fi

cd ../
echo "Done installing all dependencies."