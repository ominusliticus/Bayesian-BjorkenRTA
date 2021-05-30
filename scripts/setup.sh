#!/bin/bash

echo "Installing 3rd party libraries"
echo pwd
cd ../3rd_party

if [[-d fmt]]
then
    cd fmt
    echo pwd
    echo "Building fmt library"
    cmake .
    make
    cd ..
    echo "Done."
else
    echo "fmt repository doesn't exist"
    echo "cloning repository https://github.com/fmtlib/fmt.git"
    git clone --recursive https://github.com/fmtlib/fmt.git
    cd fmt
    echo pwd
    echo "Building fmt library"
    cmake .
    make
    cd ..
    echo "Done."
fi

cd ../scripts
echo "Done installing all dependencies."