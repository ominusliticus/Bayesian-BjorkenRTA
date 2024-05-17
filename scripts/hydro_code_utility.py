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

from typing import List, Dict
from pathlib import Path


def convert_to_PL_and_PT(
    p: np.ndarray,
    pi: np.ndarray,
    Pi: np.ndarray
) -> np.ndarray:
    '''
    Converts input shear and bulk pressure to longitudinal and transverse
    pressure
    '''
    pt = Pi + pi / 2 + p
    pl = Pi - pi + p
    return pt, pl

def read_hydro_ouput(
    hydro_name: str,
    params_dict: Dict[str, float],
    use_PL_PT: bool,
    path_to_output: Path,
) -> np.ndarray:
    '''
    Opens outputted files from C++ programs and extract output
    '''
    mass = 0.197 * params_dict['mass']  # in MeV
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
        tau, e, pi, Pi, p = f_e[i].split()[0], f_e[i].split()[1],\
                            f_pi[i].split()[1], f_Pi[i].split()[1],\
                            f_e[i].split()[2]
        if use_PL_PT:
            p1, p2 = self.convert_to_PL_and_PT(
                float(p),
                float(pi),
                float(Pi)
            )
        else:
            p1, p2 = float(pi), float(Pi)

        out_list.append([float(tau),
                         float(e),
                         float(p1),
                         float(p2),
                         float(p)])

    return np.array(out_list)

