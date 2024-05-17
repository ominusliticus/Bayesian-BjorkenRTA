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
# File: hydro_code_api.py
# Description: This file calls the executable generated from
#              the C++ routines defined in the C++ code
#              one directory higher

# For directory changing and running command line commands
from os import chdir as cd
from subprocess import run as cmd
from subprocess import CalledProcessError


class HydroCodeAPI:
    """
    This class calls the C++ program, via command line
    Constructor parameters
    ---------
    output_path - str, path that determines where to output files generated
                  from running the C++ code. Also tells python where to read
                  output from
    """

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

        try:
            cmd(['mkdir', '-p', output_path]).check_returncode()
        except (CalledProcessError):
            print(f'Failed to create dir {output_path}')
        # data slots for storing hydro runs

    def print_commandline_args(self,
                               params_dict: Dict[str, float]
                               ) -> List[str]:
        '''
        Function ouputs file "params.txt" to the Code/util folder to
        be used by the Code/build/exact_solution.x program
        '''
        keys = list(params_dict.keys())
        values = list(params_dict.values())
        return_val = f'{keys[0]} {values[0]}'
        for i in range(1, len(keys)):
            if keys[i] == 'hydro_type':
                continue
            return_val += f' {keys[i]} {values[i]}'
        return return_val.split()

    def execute_hydro_code(self,
                           params_dict: Dict[str, float],
                           which_hydro: int) -> None:
        '''
        Function calls the C++ excecutable that run hydro calculations
        '''
        cd('../')
        cmd_list = ['./build/exact_solution.x',
                    *self.print_commandline_args(params_dict),
                    f'{which_hydro}',
                    self.output_path]
        try:
            cmd(cmd_list).check_returncode()
        except (CalledProcessError):
            print("Execution off hydro code failed.\nExiting. . .\n")
        cd('scripts/')
        return None
