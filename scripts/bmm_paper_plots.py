import numpy as np
import matplotlib.pyplot as plt
import my_plotting as mp
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from typing import Dict
from typing import Union
from matplotlib.cm import plasma
from hydro_code_api import HydroCodeAPI as HCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
from my_plotting import costumize_axis
from subprocess import run as cmd


def hydro_type_from_string(hydro_name: str) -> int:
    match hydro_name:
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


def get_temp(
        energy_density: float,
        mass: float,
) -> float:
    from scipy.special import kn

    # calculate energy density given temperature and mass
    def e(temp, mass):
        z = mass / temp
        result = (z ** 2 * kn(2, z) / 2 + z ** 3 * kn(1, z) / 6)
        return 3 * temp ** 4 * result / np.pi ** 2

    # calculate equilibrium pressure given temperature and mass
    def p(temp, mass):
        z = mass / temp
        return z ** 2 * temp ** 4 * kn(2, z) / (2 * np.pi ** 2)

    # invert given energy density
    t_min = 0.001 / 0.197
    t_max = 2.0 / 0.197
    x_1 = t_min
    x_2 = t_max

    n = 0
    flag = 0
    copy = 0
    while flag != 1 and n <= 2000:
        mid = (x_1 + x_2) / 2.0
        e_mid = e(mid, mass)
        e_1 = e(x_1, mass)

        if np.abs(e_mid - energy_density) < 1e-6:
            break

        if (e_mid - energy_density) * (e_1 - e_mid) <= 0.0:
            x_2 = mid
        else:
            x_1 = mid

        n += 1
        if n == 1:
            copy = mid

        if n > 4:
            if np.abs(copy - mid) < 1e-6:
                flag = 1
            copy = mid

    return mid, p(mid, mass)


def extract_numbers_from_file(
    hydro_name: str,
    params_dict: Dict[str, Union[float, int]],
    path_to_output: Path,
    mass: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hca = HCA(path_to_output)

    if hydro_name == 'exact':
        output = hca.get_exact_results({'mass': mass})
    else:
        output = hca.get_from_output_files(
            params_dict={'hydro_type': hydro_type_from_string(hydro_name),
                         'mass': mass},
            use_PL_PT=True,
        )

    output = output.T

    tau = output[0]
    e = output[1]
    pt = output[2]
    pl = output[3]
    p = output[4]

    pi = 2 * (pt - pl) / 3
    Pi = (2 * pt + pl) / 3 - p
    h = e + p

    return np.array([tau, e, pi / h, Pi / h])


def return_residual(
        hydro: np.ndarray,
        kinetic: np.ndarray,
) -> np.ndarray:
    e_hydro = hydro[1]
    pi_hydro = hydro[2]
    Pi_hydro = hydro[3]

    e_kinetic = kinetic[1]
    pi_kinetic = kinetic[2]
    Pi_kinetic = kinetic[3]

    e_resid =  100 * np.fabs((e_hydro - e_kinetic) / e_kinetic)
    pi_resid = 100 * np.fabs((pi_hydro - pi_kinetic) / pi_kinetic
    )
    Pi_resid = 100 * np.fabs((Pi_hydro - Pi_kinetic) / Pi_kinetic)

    ret_val = np.array([e_resid, pi_resid, Pi_resid])
    ret_val[np.where(np.isnan(ret_val))] = 1.0
    return ret_val


def make_hydro_heatmap(
        params_dict: Dict[str, Union[float, str]],
        use_PL_PT: bool,
        output_dir: Path,
) -> None:
    hca = HCA(str(Path(output_dir).absolute()))
    param_names = ['C']
    hydro_names = ['ce', 'dnmr', 'mvah']

    fig, ax = plt.subplots(
            nrows=len(hydro_names),
            ncols=3,
            figsize=(3 * 7, len(hydro_names) * 7))
    fig.patch.set_facecolor('white')

    p1_name = (r'${\mathcal P_T} / \mathcal P_\mathrm{eq]$'
               if use_PL_PT else
               r'$\pi/(\mathcal E + \mathcal P_\mathrm{eq})$')
    p2_name = (r'${\mathcal P_L} / \mathcal P_\mathrm{eq]$'
               if use_PL_PT else
               r'$\Pi/(\mathcal E + \mathcal P_\mathrm{eq})$')
    col_names = [r'$\mathcal{E} / \mathcal{E}_0$', p1_name, p2_name]

    exact_output = hca.process_hydro(
        params_dict=params_dict,
        parameter_names=param_names,
        design_point=[0.39],
        use_PL_PT=use_PL_PT,
    )
    exact_output = exact_output.T

    exact_tau = exact_output[0]
    exact_e = exact_output[1]
    exact_pt = exact_output[2]
    exact_pl = exact_output[3]
    exact_p = exact_output[4]

    exact_pi = 2 * (exact_pt - exact_pl) / 3
    exact_Pi = (2 * exact_pt + exact_pl) / 3 - exact_p
    exact_h = exact_e + exact_p

    e0 = exact_e[0]
    exact_output_array = np.array([
        exact_tau,
        exact_e / e0,
        exact_pi / exact_h,
        exact_Pi / exact_h]
    ).T
    exact_tau_R = np.array([
        5 * 0.39 / get_temp(
            energy_density=exact_ee * e0,
            mass=params_dict['mass']
        )[0]
        for exact_ee in exact_output_array[..., 1].reshape(-1,)
    ])
    for i, hydro_name in enumerate(hydro_names):
        output_array = []
        for eta_s in tqdm(np.linspace(0.08, 0.80, 200)):
            params_dict['hydro_type'] = hydro_type_from_string(hydro_name)
            output = hca.process_hydro(
                params_dict=params_dict,
                parameter_names=param_names,
                design_point=[eta_s],
                use_PL_PT=use_PL_PT,
            )

            output = output.T

            tau = output[0]
            e = output[1]
            pt = output[2]
            pl = output[3]
            p = output[4]

            pi = 2 * (pt - pl) / 3
            Pi = (2 * pt + pl) / 3 - p
            h = e + p

            e0 = e[0]
            output_array.append(np.array([tau, e / e0, pi / h, Pi / h]).T)


        output_array = np.array(output_array)
        print(output_array.shape)
        for j, col_name in enumerate(col_names):
            tau_R = np.array([
                5 * eta_s / get_temp(
                    energy_density=ee * e0,
                    mass=params_dict['mass']
                )[0]
                for ee in output_array[..., 1].reshape(-1,)
            ])
            ax[i, j].hist2d(
                output_array[..., 0].reshape(-1,) / tau_R,
                output_array[..., j + 1].reshape(-1,),
                bins=100,
                cmap=plasma,
                norm='log',
                alpha=0.5,
            )
            ax[i, j].plot(
                exact_output_array[..., 0].reshape(-1,) / exact_tau_R,
                exact_output_array[..., j + 1].reshape(-1,),
                color='black',
            )
            costumize_axis(
                ax=ax[i, j],
                x_title=r'$\tau / \tau_R$',
                y_title=col_name
            )
            if j == 0:
                ax[i, j].set_yscale('log')


    plot_file = Path(f'{output_dir}/hydro_heatmaps.pdf').absolute()
    try:
        (cmd(['mkdir', '-p', str(plot_file.parent)])
            .check_returncode())
    except (CalledProcessError):
        print(f"Could not create dir {plot_file}")
    fig.tight_layout()
    fig.savefig(str(plot_file))


def make_residual_plots() -> None:
    hydro_names = ['ce', 'dnmr', 'mis', 'vah', 'mvah', 'exact']
    colors = mp.get_cmap(10, 'tab10')
    eta_S = 0.39

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(3 * 5, 1.2 * 5))
    fig.patch.set_facecolor('white')

    output_name = 'presentation_plots_2'
    output = dict(
        (
            hydro_name,
            extract_numbers_from_file(
                hydro_name=hydro_name,
                # path_to_output='../output/magenta',
                # path_to_output='../output/maroon',
                # path_to_output='../output/blue',
                # path_to_output='../output/test_no_filesystem',
                path_to_output='../output/' + output_name,
                mass=0.200 / 0.197,
            )
        )
        for hydro_name in hydro_names
    )

    for i, hydro_name in enumerate(hydro_names):
        resids = return_residual(output[hydro_name], output['exact'])
        for j, y_axis in enumerate([
            r'$\mathcal E / \mathcal E_0$',
            r'$\pi / (\mathcal E + \mathcal P_\mathrm{eq})$',
            r'$\Pi / (\mathcal E + \mathcal P_\mathrm{eq})$'
        ]):
            tau_R = np.array([
                5 * eta_S / get_temp(
                    energy_density=e,
                    mass=0.200 / 0.197
                )[0]
                for e in output[hydro_name][1]
            ])
            ax[j].plot(
                output[hydro_name][0] / tau_R,
                output[hydro_name][j + 1] / (output[hydro_name][j + 1][0] if j == 0 else 1),
                color=colors(i)
                if hydro_name != 'exact' else 'black',
                lw=2,
                label=hydro_name.upper()
                if hydro_name != 'exact' else hydro_name,
                ls='solid'
                if hydro_name != 'exact' else 'dashed',
            )
            mp.costumize_axis(
                ax=ax[j],
                x_title='',
                y_title=y_axis
            )
            ax[j].set_xscale('log')
            # ax[j].set_xlim(1, 1000)
            ax[j].set_xticks([])

    for j, y_axis in enumerate([
        r'$\mathcal E / \mathcal E_0$',
        r'$\pi / (\mathcal E + \mathcal P_\mathrm{eq})$',
        r'$\Pi / (\mathcal E + \mathcal P_\mathrm{eq})$'
    ]):
        divider = make_axes_locatable(ax[j])
        ax2 = divider.append_axes(
            "bottom",
            size="33%",
            pad=0
        )
        ax[j].figure.add_axes(ax2)

        for i, hydro_name in enumerate(hydro_names):
            if hydro_name == 'exact':
                continue
            tau_R = np.array([
                5 * eta_S / get_temp(
                    energy_density=e,
                    mass=0.200 / 0.197
                )[0] * 0.197
                for e in output[hydro_name][1]
            ])
            resids = return_residual(output[hydro_name], output['exact'])
            ax2.plot(
                output[hydro_name][0] / tau_R,
                resids[j],
                color=colors(i),
                lw=2
            )
        # ax2.set_xlim(1, 1000)
        mp.costumize_axis(
            ax=ax2,
            x_title=r'$\tau / \tau_R$',
            y_title='%'
        )
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.locator_params('y', numticks=6)

    ax[0].legend(fontsize=16)
    ax[0].set_yscale('log')
    fig.tight_layout()
    fig.savefig(f'./plots/{output_name}.pdf')


if __name__ == "__main__":
    print(get_temp(10.00, 0.200 / 0.197))
    make_hydro_heatmap(
        params_dict={
            'tau_0': 0.1,
            'e0': 12.4991,
            'pt0': 6.0977,
            'pl0': 0.0090,
            'tau_f': 12.1,
            'mass': 0.2 / 0.197,
            'C': 5 / (4 * np.pi),
            'hydro_type': 0
        },
        use_PL_PT=False,
        output_dir='../plots',
    )
