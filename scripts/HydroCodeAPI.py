#
# Author: Kevin Ingles
# File: HydroCodeAPI.py
# Description: This file calls the executable generated from
#              from the C++ routines defined in the C++ property
#              one directory higher

from platform import uname

# For directory changing and running command line commands
from os import chdir as cd, getcwd
from subprocess import run as cmd
from subprocess import CalledProcessError

# For data manipulation and generation
import numpy as np

from typing import List, Dict

# for running hydro in parallel
from multiprocessing import Manager, Process

# For progress bars 
from tqdm import tqdm

# TODO: Make GetExactSolution take the `use_PT_PL` keyword to hide this
#       complexity


class HydroCodeAPI:
    """
    Add description
    """

    def __init__(self, path_to_output: str) -> None:
        self.hydro_names = ['ce', 'dnmr', 'vah', 'mvah']
        self.path_to_output = path_to_output
        # data slots for storing hydro runs

    def PrintCommandLineArgs(self,
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

    def ExecuteHydroCode(self,
                         params_dict: Dict[str, float],
                         which_hydro: int) -> None:
        '''
        Function calls the C++ excecutable that run hydro calculations
        '''
        cd('../')
        cmd_list = ['./build/exact_solution.x',
                    *self.PrintCommandLineArgs(params_dict),
                    f'{which_hydro}',
                    self.path_to_output]
        try:
            cmd(cmd_list).check_returncode()
        except (CalledProcessError):
            print("Execution off hydro code failed.\nExiting. . .\n")
        cd('scripts/')
        return None

    def ConvertToPTandPL(self,
                         p: np.ndarray,
                         pi: np.ndarray,
                         Pi: np.ndarray) -> np.ndarray:
        pt = Pi + pi / 2 + p
        pl = Pi - pi + p
        return pt, pl

    def GetFromOutputFiles(self,
                           params_dict: Dict[str, float],
                           use_PT_PL: bool) -> np.ndarray:
        '''
        Add description
        '''
        hydro_type = params_dict['hydro_type']
        mass = 0.197 * params_dict['mass']  # in MeV

        if hydro_type == 0:
            prefix = '/ce_'
        elif hydro_type == 1:
            prefix = '/dnmr_'
        elif hydro_type == 2:
            prefix = '/vah_'
        elif hydro_type == 3:
            prefix = '/mvah_'

        f_e = open(
            self.path_to_output + prefix + 'e' + f'_m={mass:.3f}GeV.dat',
            'r'
            ).readlines()
        f_pi = open(
            self.path_to_output + prefix + 'shear' + f'_m={mass:.3f}GeV.dat',
            'r'
            ).readlines()
        f_Pi = open(
            self.path_to_output + prefix + 'bulk' + f'_m={mass:.3f}GeV.dat',
            'r'
            ).readlines()

        out_list = []
        for i in range(len(f_e)):
            tau, e, pi, Pi, p = f_e[i].split()[0], f_e[i].split()[1],\
                                f_pi[i].split()[1], f_Pi[i].split()[1],\
                                f_e[i].split()[2]
            if use_PT_PL:
                p1, p2 = self.ConvertToPTandPL(float(p), float(pi), float(Pi))
            else:
                p1, p2 = float(pi), float(Pi)

            out_list.append([float(tau),
                             float(e),
                             float(p1),
                             float(p2),
                             float(p)])

        return np.array(out_list)

    def GetExactResults(self,
                        params_dict: Dict[str, float]) -> np.ndarray:
        '''
        Add description
        '''
        with open(
                self.path_to_output
                + f'/exact_m={0.197 * params_dict["mass"]:.3f}GeV.dat',
                'r'
                ) as f_exact:
            output = np.array([[
                                float(entry)
                                for entry in line.split()]
                               for line in f_exact.readlines()])
            return output

    def ProcessHydro(self,
                     params_dict: Dict[str, float],
                     parameter_names: List[str],
                     design_point: np.ndarray,
                     use_PT_PL: bool = True) -> np.ndarray:
        '''
        Add description
        '''
        for i, name in enumerate(parameter_names):
            params_dict[name] = design_point[i]
        self.ExecuteHydroCode(params_dict, params_dict['hydro_type'])
        if params_dict['hydro_type'] == 4:
            return np.array(self.GetExactResults(params_dict))
        else:
            return np.array(self.GetFromOutputFiles(params_dict,
                                                    use_PT_PL))

    def RunHydro(self,
                 params_dict: Dict[str, float],
                 parameter_names: List[str],
                 design_points: np.ndarray,
                 simulation_taus: np.ndarray,
                 use_PT_PL: bool = True) -> None:
        # TODO: Enable using PL, PT, Pi and pi from same trainging set
        '''
        Add description
        '''

        # Multi-processing to run different hydros in sequence
        manager = Manager()
        hydro_output = manager.dict()
        for name in self.hydro_names:
            hydro_output[name] = None

        def for_multiprocessing(params_dict: Dict[str, float],
                                parameter_names: List[str],
                                design_points: np.ndarray,
                                output_dict: Dict[str, np.ndarray],
                                key: str,
                                itr: int):
            # Calculate indices for observation times
            if 'tau_0' in parameter_names:
                j = parameter_names.index('tau_0')
                observ_indices = np.array(
                    [[(tau_f / design_point[j] - 1.0) * 20.0
                      for tau_f in simulation_taus]
                     for design_point in design_points])
            else:
                tau_0 = params_dict['tau_0']
                observ_indices = np.array(
                    [[(tau_f / tau_0 - 1.0) * 20.0
                      for tau_f in simulation_taus]
                     for design_point in design_points])

            params_dict['hydro_type'] = itr
            names = ['ce', 'dnmr', 'vah', 'mvah']
            output = np.array(
                [[self.ProcessHydro(
                        params_dict,
                        parameter_names,
                        design_point,
                        use_PT_PL)[int(j)-1]
                  for j in observ_indices[i]]
                 for i, design_point in enumerate(
                     tqdm(design_points,
                          desc=f'hydro {names[itr]}: ',
                          position=itr))])
            output_dict[key] = output

        # Having trouble getting multiprocessing to work on mac
        if 'Darwin' in uname():
            for i, name in enumerate(self.hydro_names):
                for_multiprocessing(params_dict=params_dict,
                                    parameter_names=parameter_names,
                                    design_points=design_points,
                                    output_dict=hydro_output,
                                    key=name,
                                    itr=i)
        else:
            jobs = [Process(target=for_multiprocessing,
                            args=(params_dict,
                                  parameter_names,
                                  design_points,
                                  hydro_output,
                                  key,
                                  i))
                    for i, key in enumerate(self.hydro_names)]

            print("We are using linux!")
            _ = [proc.start() for proc in jobs]
            _ = [proc.join() for proc in jobs]

        for k, name in enumerate(self.hydro_names):
            for j, tau in enumerate(simulation_taus):
                with open(
                        ('{}/{}_simulation_points_n='
                         + '{}_tau={}.dat').
                        format(self.path_to_output,
                               name,
                               len(parameter_names),
                               tau),
                        'w') as f_hydro_simulation_taus:
                    for line in hydro_output[name][:, j, :]:
                        for entry in line:
                            f_hydro_simulation_taus.write(f'{entry} ')
                        f_hydro_simulation_taus.write('\n')
