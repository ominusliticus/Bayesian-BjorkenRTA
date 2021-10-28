import numpy as np
import subprocess as sp
import os

def PrintParametersFile(params_dict):
    '''
    Function ouputs file "params.txt" to the Code/util folder to be used by the
    Code/build/exact_solution.x program
    '''
    os.chdir('/mnt/c/Users/gil-c/Documents/Heinz_Research/TeX-Docs/Rough-Drafts/Bayesian-Toy-Model/Code')
    with open('./utils/params.txt', 'w') as fout:
        fout.write(f'tau_0 {params_dict["tau_0"]}\n')
        fout.write(f'Lambda_0 {params_dict["Lambda_0"]}\n')
        fout.write(f'alpha_0 {params_dict["alpha_0"]}\n')
        fout.write(f'xi_0 {params_dict["xi_0"]}\n')
        fout.write(f'ul {params_dict["tau_f"]}\n')
        fout.write(f'll {params_dict["tau_0"]}\n')
        fout.write(f'mass {params_dict["mass"]}\n')
        fout.write(f'eta_s {params_dict["eta_s"]}\n')
        fout.write(f'pl0 {params_dict["pl0"]}\n')
        fout.write(f'pt0 {params_dict["pt0"]}\n')
        fout.write(f'TYPE {params_dict["hydro_type"]}')
    os.chdir('/mnt/c/Users/gil-c/Documents/Heinz_Research/TeX-Docs/Rough-Drafts/Bayesian-Toy-Model/Code/scripts/')
    return 

def RunHydroSimulation():
    '''
    Function calls the C++ excecutable that run hydro calculations
    '''
    os.chdir('../')
    sp.run(['./build/exact_solution.x'], shell=True)
    os.chdir('scripts/')
    return

params = {
    'tau_0': 0.1,
    'Lambda_0': 1.647204044,
    'xi_0': -0.8320365099,
    'alpha_0': 0.654868759,
    'tau_f': 0.2,
    'mass': 1.015228426,
    'eta_s': 0.23873241463784,
    'pl0': 8.1705525351457684,
    'pt0': 1.9875332965147663,
    'hydro_type': 4
}

# function run runs hydro code to generate simualtion results for a set of 
# given parameters
def ProcessHydro(parameter_names, simulation_points):
    out_list = []
    def GetExactResults():
        with open('../output/exact/MCMC_calculation_moments.dat','r') as f_exact:
            t, e, pl, pt, p = f_exact.readlines()[0].split()
            temp_list = [float(t), float(e), float(p), float(pt), float(pl)]
            return temp_list

    if len(simulation_points) > len(parameter_names):
        for parameters in simulation_points:
            for i, name in enumerate(parameter_names):
                params[name] = parameters[i]
            PrintParametersFile(params)
            RunHydroSimulation()
            out_list.append(GetExactResults())

    return np.array(out_list)


if __name__ == '__main__':
    xis = np.linspace(-0.999, 10, 1000).reshape(-1,1)
    output = ProcessHydro(['xi_0'], xis)

    for C in [1 / (4 * np.pi), 10 / (4 * np.pi)]:
        params['eta_s'] = C
        with open(f'../output/initial_values_C={C:.3f}.dat','w') as f:
            for i, line in enumerate(output):
                for entry in line:
                    f.write(f'{entry} ')
                f.write(f'{xis[i, 0]}\n') 