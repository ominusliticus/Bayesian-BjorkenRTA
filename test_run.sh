#!/bin/bash
set -x

mkdir -p output/test
make 
./build/exact_solution.x tau_0 0.1 e0 12.0 pt 3.0 pl0 2.0 tau_f 12.1 mass 1.01523 C 0.08 $1 "output/mvah_debug"
