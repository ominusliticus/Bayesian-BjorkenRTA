#!/bin/bash
set -x

mkdir -p output/test
make 
./build/exact_solution.x tau_0 0.1 e0 12.4991 pt0 6.0977 pl0 4.000 tau_f 100.1 mass 1.01523 C 0.08 $1 "output/test"
