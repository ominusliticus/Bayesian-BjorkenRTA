#  Copyright 2021-2024 Kevin Ingles
#
#  Permission is hereby granted, free of charge, to any person obtaining
#  a copy of this software and associated documentation files (the
#  "Software"), to deal in the Software without restriction, including
#  without limitation the right to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of the Software, and to
#  permit persons to whom the Sofware is furnished to do so, subject to
#  the following conditions:
#
#  The above copyright notice and this permission notice shall be
#  included in all copies or substantial poritions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#  SOFTWARE OR THE USE OF OTHER DEALINGS IN THE SOFTWARE
#
# Author: Kevin Ingles
# File: hydro_code_utility.py
# Description: Interface to facilitate reading in files outputted from running
#              C++ hydro code

from hydro_code_cmdln_options import HydroCodeCmdOptions
from pathlib import Path
import numpy as np


def convert_hydro_name_to_int(name: str) -> int:
    '''
    Returns integer corresponding to the hydro in C++ code.
    See documentation or ../src/main.cpp: int main() for options
    '''
    match name:
        case 'ce':
            return 0
        case 'dnmr':
            return 1
        case 'mis':
            return 2
        case 'vah':
            return 3
        case 'mvah':
            return 4
        case 'exact':
            return 5


def convert_int_to_hydro_name(n: int) -> str:
    '''
    Take integer corresponding to the hydro in C++ code and returns name.
    See documentation or ../src/main.cpp: int main() for options
    '''
    match n:
        case 0:
            return 'ce'
        case 1:
            return 'dnmr'
        case 2:
            return 'mis'
        case 3:
            return 'vah'
        case 4:
            return 'mvah'
        case 5:
            return 'exact'


def read_hydro_ouput(
    cmdln_options: HydroCodeCmdOptions,
    path_to_output: Path,
) -> np.ndarray:
    '''
    Opens outputted files from C++ programs and extract output
    '''
    path_to_output = path_to_output / "code_output"
    if cmdln_options['hydro_type'] == 5:  # Run exact hydro
        with open(
                path_to_output,
                + f'/exact_m={0.197 * cmdln_options["mass"]:.3f}GeV.dat',
                'r'
                ) as f_exact:
            output = np.array([[float(entry)
                                for entry in line.split()]
                               for line in f_exact.readlines()])

        out_list = []
        for entry in output:
            tau, e, pt, pl, p = entry
            shear = (2.0 / 3.0) * (pt - pl)
            bulk = (pl + 2.0 * pt) / 3.0 - p

            out_list.append([tau, e, shear, bulk, p])

            return np.array(out_list)
    else:
        hydro_name = convert_int_to_hydro_name(n=cmdln_options['hydro_type'])
        mass = 0.197 * cmdln_options['mass']  # in MeV
        prefix = '/' + hydro_name + '_'

        f_e = open(
            path_to_output + prefix + 'e' + f'_m={mass:.3f}GeV.dat',
            'r'
            ).readlines()
        f_pi = open(
            path_to_output + prefix + 'shear' + f'_m={mass:.3f}GeV.dat',
            'r'
            ).readlines()
        f_Pi = open(
            path_to_output + prefix + 'bulk' + f'_m={mass:.3f}GeV.dat',
            'r'
            ).readlines()

        out_list = []
        for i in range(len(f_e)):
            tau, e, shear, bulk, p = f_e[i].split()[0], f_e[i].split()[1], \
                                     f_pi[i].split()[1], f_Pi[i].split()[1], \
                                     f_e[i].split()[2]
            out_list.append([float(tau),
                             float(e),
                             float(shear),
                             float(bulk),
                             float(p)])

    return np.array(out_list)
