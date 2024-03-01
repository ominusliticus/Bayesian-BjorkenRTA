#!/bin/bash
set -x

# ouput_dir='output/presentation_plots_1'
# mkdir -p $ouput_dir
# make 
# for i in `seq 0 5`;
# do
#     ./build/exact_solution.x tau_0 0.1 e0 12.4625 pt0 1.99149 pl0 8.18535 tau_f 100.1 mass 1.01523 C 0.39 $i $ouput_dir
# done

# ouput_dir='output/presentation_plots_2'
# mkdir -p $ouput_dir
# make 
# for i in `seq 0 5`;
# do
#     ./build/exact_solution.x tau_0 0.1 e0 10.000 pt0 1.000 pl0 2.000 tau_f 100.1 mass 1.01523 C 0.39 $i $ouput_dir
# done

# ouput_dir='output/presentation_plots_3'
# mkdir -p $ouput_dir
# make 
# for i in `seq 0 5`; do
#     ./build/exact_solution.x tau_0 0.1 e0 10.000 pt0 2.000 pl0 0.500 tau_f 100.1 mass 1.01523 C 0.39 $i $ouput_dir
# done

ouput_dir='output/presentation_plots_4'
mkdir -p $ouput_dir
make 
for i in `seq 0 5`;
do
    ./build/exact_solution.x tau_0 0.1 e0 100.000 pt0 1.000 pl0 10.000 tau_f 100.1 mass 1.01523 C 0.39 $i $ouput_dir
done