#!/bin/bash

mkdir -p output/test
make 
echo "./build/exact_solution.x tau_0 0.1 e0 12.4991 pt0 6.0977 pl0 0.0090 tau_f 12.1 mass 1.01523 C 0.079 $1 \"output/test\""
./build/exact_solution.x tau_0 0.1 e0 12.4991 pt0 6.0977 pl0 0.0090 tau_f 12.1 mass 1.01523 C 0.08 $1 "output/test"
