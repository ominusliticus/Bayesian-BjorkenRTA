#!/bin/bash
set -x

mkdir -p output/maroon
make 
./build/exact_solution.x tau_0 0.1 e0 12.4625 pt0 1.99149 pl0 8.18535 tau_f 12.1 mass 1.01523 C 0.39 $1 "output/test_exact"
