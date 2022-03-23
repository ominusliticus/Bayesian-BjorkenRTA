#
# Author: Kevin Ingles
# File: HydroCodeAPI.py
# Description: This file calls the executable generated from
#              from the C++ routines defined in the C++ property
#              one directory higher

# For directory changing and running command line commands
from os import chdir as cd
from subprocess import run as cmd

# For data manipulation and generation
import numpy as np

# for storing and reading arrays easily
import pickle

from types import List, Dict


class HydroCodeAPI:
    """
    Add description
    """

    def __init__(self, default_params_dict: Dict) -> None:
        self.hydro_names = ['ce', 'dnmr', 'vah', 'mvah']
        self.params = default_params_dict

        # data slots for storing hydro runs
        self.hydro_outputs_dict

    def PrintParametersFile(self, params_dict: Dict) -> None:
        '''
        Function ouputs file "params.txt" to the Code/util folder to
        be used by the Code/build/exact_solution.x program
        '''
        cd('../')
        with open('./utils/params.txt', 'w') as fout:
            fout.write(f'tau_0 {params_dict["tau_0"]}\n')
            fout.write(f'Lambda_0 {params_dict["Lambda_0"]}\n')
            fout.write(f'alpha_0 {params_dict["alpha_0"]}\n')
            fout.write(f'xi_0 {params_dict["xi_0"]}\n')
            fout.write(f'ul {params_dict["tau_f"]}\n')
            fout.write(f'll {params_dict["tau_0"]}\n')
            fout.write(f'mass {params_dict["mass"]}\n')
            fout.write(f'eta_s {params_dict["eta_s"]}\n')
            fout.write(f'TYPE {params_dict["hydro_type"]}')
        cd('scripts/')
        return None

    def RunHydroSimulation(self) -> None:
        '''
        Function calls the C++ excecutable that run hydro calculations
        '''
        cd('../')
        cmd(['./build/exact_solution.x'], shell=True)
        cd('scripts/')
        return None

    def ProcessHydro(self, 
                     parameter_names: List,
                     simulation_points: List,
                     store_whole_file: bool = False) -> np.ndarray:
        out_list = []

        def ConvertToPTandPL(p: np.ndarray,
                             pi: np.ndarray,
                             Pi: np.ndarray) -> np.ndarray:
            pt = Pi + pi / 2 + p
            pl = Pi - pi + p
            return pt, pl

        def GetFromOutputFiles(hydro_type: str) -> np.ndarray:
            if hydro_type == 0:
                prefix = '../output/CE_hydro/'
                suffix = ''
            elif hydro_type == 1:
                prefix = '../output/DNMR_hydro/'
                suffix = ''
            elif hydro_type == 2:
                prefix = '../output/aniso_hydro/'
                suffix = ''
            elif hydro_type == 3:
                prefix = '../output/aniso_hydro/'
                suffix = '2'

            if store_whole_file:
                f_e = open(
                    prefix + 'e' + suffix + '_m=0.200GeV.dat', 'r').readlines()
                f_pi = open(
                    prefix + 'shear' + suffix + '_m=0.200GeV.dat', 'r'
                    ).readlines()
                f_Pi = open(
                    prefix + 'bulk' + suffix + '_m=0.200GeV.dat', 'r'
                    ).readlines()

                out_list = []
                for i in range(len(f_e)):
                    tau, e, pi, Pi, p = f_e[i].split()[0], f_e[i].split()[1],\
                                        f_pi[i].split()[1], f_Pi[i].split()[1],\
                                        f_e[i].split()[2]
                    pt, pl = ConvertToPTandPL(float(p), float(pi), float(Pi))
                    out_list.append([float(tau),
                                     float(e),
                                     float(pt),
                                     float(pl),
                                     float(p)])

                return np.array(out_list)
            else:
                # TODO: update this to return PT and PL instead of pi and Pi
                f_e = open(prefix + 'e' + suffix + '_m=0.200GeV.dat', 'r')
                last_e = f_e.readlines()[-1]
                tau, e = last_e.split()[0], last_e.split()[1]
                f_e.close()
                del last_e

                f_pi = open(prefix + 'shear' + suffix + '_m=0.200GeV.dat', 'r')
                last_pi = f_pi.readlines()[-1]
                pi = last_pi.split()[1]
                f_pi.close()
                del last_pi

                f_Pi = open(prefix + 'bulk' + suffix + '_m=0.200GeV.dat', 'r')
                last_Pi = f_Pi.readlines()[-1]
                Pi = last_Pi.split()[1]
                f_Pi.close()
                del last_Pi

                temp_list = [float(tau), float(e), float(pi), float(Pi)]
                return np.array(temp_list)

        def GetExactResults() -> np.ndarray:
            with open(
                    '../output/exact/MCMC_calculation_moments.dat', 'r'
                    ) as f_exact:
                if store_whole_file:
                    output = np.array([[
                                        float(entry)
                                        for entry in line.split()]
                                       for line in f_exact.readlines()])
                    return output
                else:
                    return np.array([float(entry) 
                                     for entry in 
                                     f_exact.readlines()[-1].split()])

        if len(simulation_points) > len(parameter_names):
            for parameters in simulation_points:
                for i, name in enumerate(parameter_names):
                    self.params[name] = parameters[i]
                self.PrintParametersFile(self.params)
                self.RunHydroSimulation()
                if self.params['hydro_type'] == 4:
                    return np.array(GetExactResults())
                else:
                    out_list.append(
                        GetFromOutputFiles(self.params['hydro_type']))

        else:
            for i, name in enumerate(parameter_names):
                self.params[name] = simulation_points[i]
            self.PrintParametersFile(self.params)
            self.RunHydroSimulation()
            if self.params['hydro_type'] == 4:
                return np.array(GetExactResults())
            else:
                return np.array(GetFromOutputFiles(self.params['hydro_type']))

        return np.array(out_list)
