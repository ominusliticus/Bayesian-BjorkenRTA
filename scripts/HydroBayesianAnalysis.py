# For changing directories to C++ programming and runnning files
import subprocess as sp
import os
from typing import Dict, List

# Typical functionality for data manipulation and generation of latin hypercube
import numpy as np
from pyDOE import lhs
import ptemcee

# Gaussian Process emulator 
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl

# For calculations
from scipy.linalg import lapack

# data storage
import pickle

# for warnings when running on WSL
os.environ['MPLCONFIGDIR'] = '/tmp/'


class HydroBayesianAnalysis(object):
    """
    Add description
    """
    def __init__(
            self,
            default_params: Dict,
            parameter_names: List,
            parameter_ranges: np.ndarray,
            simulation_taus: np.ndarray,
            run_new_hydro: bool,
            train_GP: bool
            ) -> None:
        print("Initializing Bayesian Analysis class")
        self.hydro_names = ['ce', 'dnmr', 'vah', 'mvah']
        self.params = default_params 
        self.parameter_names = parameter_names
        self.num_params = len(parameter_names)
        self.parameter_ranges = parameter_ranges
        self.simulation_taus = simulation_taus
        self.train_GP = train_GP
        self.run_new_hydro = run_new_hydro

        self.MCMC_chains = {}   # Dict[str, np.ndarray]
        self.evidence = {}      # Dict[str, float]
        
        if not self.train_GP:
            # Load GP and scalers data from pickle files
            with open(f'design_points/design_points_n={self.num_params}.dat','r') as f:
                self.design_points = np.array([[float(entry) for entry in line.split()] for line in f.readlines()])
            f_pickle_emulators = open(f'pickle_files/emulators_data_n={self.num_params}.pkl','rb')
            self.GP_emulators = pickle.load(f_pickle_emulators)
            self.scalers = {}
            f_pickle_emulators.close()
        else:
            print("Running hydro")
            # Run hydro code and generate scalers and GP pickle files
            unit = lhs(n=self.num_params, samples=20 * self.num_params, criterion='maximin')
            self.design_points = self.parameter_ranges[:,0] + unit * (self.parameter_ranges[:,1] - self.parameter_ranges[:,0])
            design_points = self.design_points

            global_last_output = dict((key, []) for key in self.hydro_names)
            global_full_output = dict((key, []) for key in self.hydro_names)
            hydro_simulations = dict((key, []) for key in self.hydro_names)

            if run_new_hydro: 
                tau_start = self.params['tau_0']
                delta_tau = tau_start / 20
                n_steps_1_fm = 1 / delta_tau
                self.params['tau_f'] = simulation_taus[-1]

                for i in range(4): # index for hydro names
                    self.params['hydro_type'] = i
                    for design_point in design_points:
                        local_last_output = []
                        hydro_output = self.ProcessHydro(parameter_names=self.parameter_names, simulation_points=design_point, store_whole_file=True)
                        for j in np.arange(int(simulation_taus[0]), int(simulation_taus[-1])+1, 1):
                            local_last_output.append(hydro_output[int(j * n_steps_1_fm) - 1, :])
                        global_last_output[self.hydro_names[i]].append(local_last_output)
                        global_full_output[self.hydro_names[i]].append(hydro_output)   

                print(os.getcwd())
                with open(f'design_points/design_points_n={self.num_params}.dat','w') as f_design_points:
                    for line in design_points:
                        for entry in line:
                            f_design_points.write(f'{entry} ')
                        f_design_points.write(f'\n')

                for k, name in enumerate(self.hydro_names):
                    for j, tau in enumerate(simulation_taus):
                        with open(f'hydro_simulation_points/{name}_simulation_points_n={self.num_params}_tau={tau}.dat', 'w') as f_hydro_simulation_taus:
                            for line in np.array(global_last_output[name])[:, j, :]:
                                for entry in line:
                                    f_hydro_simulation_taus.write(f'{entry} ')
                                f_hydro_simulation_taus.write('\n')

                for k, name in enumerate(self.hydro_names):
                    for i, design_point in enumerate(design_points):
                        with open(f'full_outputs/{name}_full_output_C={design_point}.dat', 'w') as f_full_output:
                            for line in np.array(global_full_output[name])[i]:
                                for entry in line:
                                    f_full_output.write(f'{entry} ')
                                f_full_output.write('\n')
                                
                hydro_simulations = dict((key, np.array([np.array(global_last_output[key])[:, j, :] for j in range(simulation_taus.size)])) for key in self.hydro_names)
            else:
                with open(f'design_points/design_points_n={self.num_params}.dat', 'r') as f_design_points:
                    design_points = np.array([float(line) for line in f_design_points.readlines()])

                for k, name in enumerate(self.hydro_names):
                    for j, tau in enumerate(simulation_taus):
                        with open(f'hydro_simulation_points/{name}_simulation_points_n={self.num_params}_tau={tau}.dat', 'r') as f_hydro_simulation_points:
                            hydro_simulations[name].append([[float(entry) for entry in line.split()] for line  in f_hydro_simulation_points.readlines()])
                hydro_simulations = dict((key, np.array(hydro_simulations[key])) for key in hydro_simulations)
            
            print("Fitting emulators")
            hydro_lists = np.array([hydro_simulations[key] for key in self.hydro_names])
            

            self.GP_emulators = dict((key, []) for key in self.hydro_names)

            obs = ['E','pi','Pi']
            f_emulator_scores = open(f'full_outputs/emulator_scores_n={self.num_params}.txt', 'w')
            f_pickle_emulators = open(f'pickle_files/emulators_data_n={self.num_params}.pkl', 'wb')
            for i, name in enumerate(self.hydro_names):
                global_emulators = []
                for j, tau in enumerate(simulation_taus):
                    local_emulators = []
                    f_emulator_scores.write(f'\tTraining GP for {name}\n')
                    for m in range(1, 4):
                        data = hydro_lists[i,j,:,m].reshape(-1,1)

                        bounds = np.outer(np.diff(self.parameter_ranges), (1e-2, 1e2))
                        kernel = 1 * krnl.RBF(length_scale=np.diff(self.parameter_ranges), length_scale_bounds=bounds)
                        GPR = gpr(kernel=kernel, n_restarts_optimizer=40, alpha=1e-8, normalize_y=True)
                        f_emulator_scores.write(f'\t\tTraining GP for {name} and time {tau}\n')
                        GPR.fit(design_points.reshape(-1,self.num_params), data)

                        f_emulator_scores.write(f'Runnig fit for {name} at time {tau} fm/c for observable {obs[m-1]}')
                        f_emulator_scores.write('GP score: {:1.3f}'.format(GPR.score(design_points.reshape(-1,self.num_params), data)))
                        f_emulator_scores.write('GP parameters: {}'.format(GPR.kernel_))
                        f_emulator_scores.write('GP log-likelihood: {}'.format(GPR.log_marginal_likelihood(GPR.kernel_.theta)))
                        f_emulator_scores.write('------------------------------\n')
                        local_emulators.append(GPR)
                    global_emulators.append(local_emulators)
                self.scalers = {}
                self.GP_emulators[name] = global_emulators
            pickle.dump(self.GP_emulators, f_pickle_emulators)

            f_emulator_scores.close()
            f_pickle_emulators.close()

            print("Done")


    def LogPrior(self, evaluation_point: np.ndarray, parameter_ranges: np.ndarray) -> float:
        '''
        Parameters:
        ------------
        evaluation_points    - 1d-array with shape (n,m). Value of parameters used to evaluate model \n
        design_range        - 2d-array with shape (m,2). Give upper and lower limits of parameter values

        where m = number of parameters in inference
        '''
        X  = np.array(evaluation_point).reshape(1,-1)
        lower = np.all(X >= np.array(parameter_ranges)[:,0])
        upper = np.all(X <= np.array(parameter_ranges)[:,1])
        
        if (np.all(lower) and np.all(upper)):
            return 0
        else:
            return -np.inf

    
    def LogLikelihood(self, evaluation_point: np.ndarray, true_observables: np.ndarray, true_errors: np.ndarray, hydro_name: str, GP_emulator: Dict, scalers: Dict={}) -> np.ndarray:
        '''
        Parameters:
        ------------
        evaluation_points    - 1d-array like (n,m) \n
        true_observables     - data
        true_error           - data error  
        hydro_name           - string containing hydro theory: 'ce', 'dnmr', 'vah', 'mvah'  \n
        GP_emulator          - dictionary(hydro_name: emulator_list), \n
                                emulator_list[0] - energery density \n
                                emulator_list[1] - shear stress  \n
                                emulator_list[2] - bulk stress
        '''
        def PredictObservable(evaluation_points: np.ndarray, hydro_name: str, tau_index: int, GP_emulator: Dict, scalers: Dict={}) -> np.ndarray:
            """
            Add description
            """
            means = []
            variances = []
            for i in range(3):
                prediction, error = GP_emulator[hydro_name][tau_index][i].predict(np.array(evaluation_points).reshape(-1, len(evaluation_points)), return_std=True)
                
                mean = prediction.reshape(-1,1)
                std = error.reshape(-1,)

                means.append(mean)
                variances.append(std ** 2)
            return np.hstack(means), np.diag(np.array(variances).flatten())

        running_log_likelihood = 0
        for k in range(true_observables.shape[0]):
            emulation_values, emulation_variance = PredictObservable(evaluation_point, hydro_name, k, GP_emulator, scalers) 

            y = np.array(emulation_values).flatten() - np.array(true_observables[k]).flatten()
            cov = emulation_variance + np.diag(true_errors[k].flatten()) ** 2

            # Use Cholesky decomposition for efficient lin alg algo
            L, info = lapack.dpotrf(cov, clean=True)

            if (info < 0):
                raise print('Error occured in computation of Cholesky decomposition')

            # Solve equation L*b=y
            b, info = lapack.dpotrs(L, np.array(y))

            if (info != 0):
                raise print('Error in inverting matrix equation')

            if np.all(L.diagonal() > 0):
                running_log_likelihood += -0.5 * np.dot(y, b) - np.log(L.diagonal()).sum()
            else:
                raise print('Diagonal has negative entry')
        
        return running_log_likelihood


    def RunMCMC(self, nsteps: int, nburn: int, ntemps: int, exact_observables: np.ndarray, exact_error: np.ndarray, read_from_file=False) -> Dict:
        """
        Parameters:
        --------------
        nsteps : Number of steps for MCMC chain to take\n
        nburn : Number of burn-in steps to take\n
        exact_observables : (n,4) np.ndarray where n is the number of simulation_taus passed at initialization\n
        exact_error : (n,3) np.ndarray where n is the number of simulation_taus passed at initialization

        Returns:
        ----------
        Dictionary of MCMC chains, index by the hydro name.\n
        MCMC chain has the shape (ntemps, nwalkers, nsteps, num_params), where nwalkers = 20 * num_params
        """
        if read_from_file:
            print("Reading mcmc_chain from file")
            with open('pickle_files/mcmc_chains.pkl','rb') as f:
                self.MCMC_chains = pickle.load(f)
            with open('pickle_files/evidence.pkl','rb') as f:
                self.evidence = pickle.load(f)
        else:
            nwalkers = 20 * self.num_params

            # TO DO: print MCMC to pickle file
            for i, name in enumerate(self.hydro_names):
                print(f"Computing for hydro theory: {name}")
                starting_guess = np.array([self.parameter_ranges[:,0] + np.random.rand(nwalkers, self.num_params) * np.diff(self.parameter_ranges) for _ in range(ntemps)])
                sampler = ptemcee.Sampler(nwalkers=nwalkers, dim=self.num_params, ntemps=ntemps, threads=4,
                    logl=self.LogLikelihood, logp=self.LogPrior,
                    loglargs=[exact_observables[:, 1:4], exact_error, name, self.GP_emulators, self.scalers],
                    logpargs=[self.parameter_ranges]
                )

                print('burn in sampling started')    
                x = sampler.run_mcmc(p0=starting_guess, iterations=nburn, swap_ratios=True)
                print("Mean acceptance fractions (in total {0} steps): ".format(ntemps*nwalkers*nburn))
                print(x[3])
                print('Burn in completed.')

                sampler.reset()

                print("Now running the samples")
                x = sampler.run_mcmc(p0=x[0], iterations=nsteps, storechain=True, swap_ratios=True)
                print("Mean acceptance fractions (in total {0} steps): ".format(ntemps*nwalkers*nsteps))
                print(x[3])
                
                self.MCMC_chains[name] = np.array(sampler.chain)
                self.evidence[name] = sampler.log_evidence_estimate()
            
            with open('pickle_files/mcmc_chains.pkl', 'wb') as f:
                pickle.dump(self.MCMC_chains, f)
            with open('pickle_files/evidence.pkl', 'wb') as f:
                pickle.dump(self.evidence, f)


    def CalculateBayesFactor(self, hydro1: str, hydro2: str) -> float:
        """
        Parameters: 
        -------------
        hydro1: name of 1 of four hydro theory being compared
        hydro2: name of a different hydro theory being compared

        Returns:
        -------------
        The Bayes factor, or ratio of evidences, for hydro1 to hydro2
        """
        return self.evidence[hydro1][0] / self.evidence[hydro2][0]


    def PrintParametersFile(self, params_dict: Dict) -> None:
        '''
        Function ouputs file "params.txt" to the Code/util folder to be used by the
        Code/build/exact_solution.x program
        '''
        os.chdir('../')
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
        os.chdir('scripts/')
        return None

    def RunHydroSimulation(self) -> None:
        '''
        Function calls the C++ excecutable that run hydro calculations
        '''
        os.chdir('../')
        sp.run(['./build/exact_solution.x'], shell=True)
        os.chdir('scripts/')
        return None

    def ProcessHydro(self, parameter_names: List, simulation_points: List, store_whole_file=False) -> np.ndarray:
        out_list = []
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
                f_e = open(prefix + 'e' + suffix + '_m=0.200GeV.dat', 'r').readlines()
                f_pi = open(prefix + 'shear' + suffix + '_m=0.200GeV.dat', 'r').readlines()
                f_Pi = open(prefix + 'bulk' + suffix + '_m=0.200GeV.dat', 'r').readlines()
                
                out_list = []
                for i in range(len(f_e)):
                    tau, e, pi, Pi, p = f_e[i].split()[0], f_e[i].split()[1], f_pi[i].split()[1], f_Pi[i].split()[1], f_e[i].split()[2]
                    out_list.append([float(tau), float(e), float(pi), float(Pi), float(p)])

                return np.array(out_list)
            else:
                f_e = open(prefix + 'e' + suffix + '_m=0.200GeV.dat', 'r')
                last_e = f_e.readlines()[-1]
                tau, e = last_e.split()[0], last_e.split()[1]
                f_e.close(); del last_e

                f_pi = open(prefix + 'shear' + suffix + '_m=0.200GeV.dat', 'r')
                last_pi = f_pi.readlines()[-1]
                pi = last_pi.split()[1]
                f_pi.close(); del last_pi

                f_Pi = open(prefix + 'bulk' + suffix + '_m=0.200GeV.dat', 'r')
                last_Pi = f_Pi.readlines()[-1]
                Pi = last_Pi.split()[1]
                f_Pi.close(); del last_Pi

                temp_list = [float(tau), float(e), float(pi), float(Pi)]
                return np.array(temp_list)

        def GetExactResults() -> List:
            with open('../output/exact/MCMC_calculation_moments.dat','r') as f_exact:
                if store_whole_file:
                    return np.array([[float(entry) for entry in line.split()] for line in f_exact.readlines()])
                else:
                    t, e, pl, pt, p = f_exact.readlines()[-1].split()
                    pi = (float(pt) - float(pl)) / 1.5
                    Pi = (2 *  float(pt) + float(pl)) / 3 - float(p)
                    temp_list = [float(t), float(e), pi, Pi, float(p)]
                    return temp_list

        if len(simulation_points) > len(parameter_names):
            for parameters in simulation_points:
                for i, name in enumerate(parameter_names):
                    self.params[name] = parameters[i]
                self.PrintParametersFile(self.params)
                self.RunHydroSimulation()
                if self.params['hydro_type'] == 4:
                    out_list.append(GetExactResults())
                else:
                    out_list.append(GetFromOutputFiles(self.params['hydro_type']))

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

    def RunExactHydroForGPDesignPoints(self):

        self.params['hydro_type'] = 4
        exact_hydro_output = np.array([
            self.ProcessHydro(parameter_names=self.parameter_names, simulation_points=design_point, store_whole_file=True)
            for design_point in self.design_points]
        )
        
        for i, design_point in enumerate(self.design_points):
            with open(f'full_outputs/exact_hydro_C={design_point}.dat','w') as f:
                for line in exact_hydro_output[i]:
                    for entry in line:
                        f.write(f'{entry} ')
                    f.write('\n')

