#
# Author: Kevin Ingles
# File: HydroEmulation.py
# Description: This file constructs the Gaussian emulators from hydro 
#              simulations for efficient code evaluation

# For choosing training data
from pyDOE import lhs

# Gaussian Process emulator
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl

'''
        if not self.train_GP:
            # Load GP and scalers data from pickle files
            with open(
                    f'design_points/design_points_n={self.num_params}.dat',
                    'r'
                    ) as f:
                self.design_points = np.array(
                    [[float(entry) for entry in line.split()]
                        for line in f.readlines()])

            f_pickle_emulators = open(
                f'pickle_files/emulators_data_n={self.num_params}.pkl',
                'rb')

            self.GP_emulators = pickle.load(f_pickle_emulators)
            self.scalers = {}
            f_pickle_emulators.close()
        else:
            print("Running hydro")
            # Run hydro code and generate scalers and GP pickle files
            unit = lhs(n=self.num_params,
                       samples=20 * self.num_params,
                       criterion='maximin')
            self.design_points = self.parameter_ranges[:, 0] + unit * \
                (self.parameter_ranges[:, 1] - self.parameter_ranges[:, 0])

            design_points = self.design_points

            global_last_output = dict((key, []) for key in self.hydro_names)
            global_full_output = dict((key, []) for key in self.hydro_names)
            hydro_simulations = dict((key, []) for key in self.hydro_names)

            if run_new_hydro:
                self.params['tau_f'] = simulation_taus[-1]

                for i in range(4):  # index for hydro names
                    self.params['hydro_type'] = i
                    for design_point in design_points:
                        local_last_output = []
                        hydro_output = self.ProcessHydro(
                            parameter_names=self.parameter_names,
                            simulation_points=design_point,
                            store_whole_file=True)

                        # By now self.params has updated with correct tau_0
                        tau_start = self.params['tau_0']
                        delta_tau = tau_start / 20
                        observ_indices = \
                            (simulation_taus -
                             np.full_like(simulation_taus,
                                          tau_start)) / delta_tau
                        for j in observ_indices:
                            local_last_output.append(
                                hydro_output[int(j) - 1, :])

                        global_last_output[
                            self.hydro_names[i]].append(local_last_output)
                        global_full_output[
                            self.hydro_names[i]].append(hydro_output)

                print(os.getcwd())
                with open(
                        f'design_points/design_points_n={self.num_params}.dat',
                        'w'
                        ) as f_design_points:
                    for line in design_points:
                        for entry in line:
                            f_design_points.write(f'{entry} ')
                        f_design_points.write('\n')

                for k, name in enumerate(self.hydro_names):
                    for j, tau in enumerate(simulation_taus):
                        with open(
                                f'''hydro_simulation_points/
                                    {name}_simulation_points_n=
                                    {self.num_params}_tau=
                                    {tau}.dat''',
                                'w'
                                ) as f_hydro_simulation_taus:
                            for line in np.array(
                                    global_last_output[name])[:, j, :]:
                                for entry in line:
                                    f_hydro_simulation_taus.write(f'{entry} ')
                                f_hydro_simulation_taus.write('\n')

                for k, name in enumerate(self.hydro_names):
                    for i, design_point in enumerate(design_points):
                        with open(
                                f'''full_outputs/
                                    {name}_full_output_C=
                                    {design_point}.dat''',
                                'w'
                                ) as f_full_output:
                            for line in np.array(global_full_output[name])[i]:
                                for entry in line:
                                    f_full_output.write(f'{entry} ')
                                f_full_output.write('\n')

                hydro_simulations = dict(
                    (key, np.array([np.array(global_last_output[key])[:, j, :]
                                    for j in range(simulation_taus.size)]))
                    for key in self.hydro_names)
            else:
                with open(
                        f'''design_points/
                        design_points_n={self.num_params}.dat''',
                        'r'
                        ) as f_design_points:
                    design_points = np.array(
                        [float(line)
                         for line in f_design_points.readlines()])

                for k, name in enumerate(self.hydro_names):
                    for j, tau in enumerate(simulation_taus):
                        with open(
                                f'''hydro_simulation_points/
                                {name}_simulation_points_n=
                                {self.num_params}_tau={tau}.dat''',
                                'r'
                                ) as f_hydro_simulation_pts:
                            hydro_simulations[name].append(
                                [[float(entry)
                                  for entry in line.split()]
                                 for line in f_hydro_simulation_pts.readlines()
                                 ])
                hydro_simulations = dict(
                    (key, np.array(hydro_simulations[key]))
                    for key in hydro_simulations)

            print("Fitting emulators")
            hydro_lists = np.array(
                [hydro_simulations[key] for key in self.hydro_names])

            self.GP_emulators = dict((key, []) for key in self.hydro_names)

            obs = ['E', 'pi', 'Pi']
            f_emulator_scores = open(
                f'full_outputs/emulator_scores_n={self.num_params}.txt', 'w')
            f_pickle_emulators = open(
                f'pickle_files/emulators_data_n={self.num_params}.pkl', 'wb')

            for i, name in enumerate(self.hydro_names):
                global_emulators = []
                for j, tau in enumerate(simulation_taus):
                    local_emulators = []
                    f_emulator_scores.write(f'\tTraining GP for {name}\n')
                    for m in range(1, 4):
                        data = hydro_lists[i, j, :, m].reshape(-1, 1)

                        bounds = np.outer(
                            np.diff(self.parameter_ranges), (1e-2, 1e2))
                        kernel = 1 * krnl.RBF(
                                length_scale=np.diff(self.parameter_ranges),
                                length_scale_bounds=bounds)
                        GPR = gpr(kernel=kernel,
                                  n_restarts_optimizer=40,
                                  alpha=1e-8,
                                  normalize_y=True)
                        f_emulator_scores.write(
                            f'\t\tTraining GP for {name} and time {tau}\n')
                        GPR.fit(design_points.reshape(-1,
                                                      self.num_params), data)

                        f_emulator_scores.write(
                            f'''Runnig fit for {name} at time {tau} fm/c
                                for observable {obs[m-1]}''')
                        f_emulator_scores.write(
                            'GP score: {:1.3f}'.format(
                                GPR.score(
                                    design_points.reshape(-1,
                                                          self.num_params),
                                    data))
                            )
                        f_emulator_scores.write(
                            'GP parameters: {}'.format(GPR.kernel_))
                        f_emulator_scores.write(
                            'GP log-likelihood: {}'.format(
                                GPR.log_marginal_likelihood(GPR.kernel_.theta))
                            )
                        f_emulator_scores.write(
                            '------------------------------\n')
                        local_emulators.append(GPR)
                    global_emulators.append(local_emulators)
                self.scalers = {}
                self.GP_emulators[name] = global_emulators
            pickle.dump(self.GP_emulators, f_pickle_emulators)

            f_emulator_scores.close()
            f_pickle_emulators.close()

            print("Done")
'''
