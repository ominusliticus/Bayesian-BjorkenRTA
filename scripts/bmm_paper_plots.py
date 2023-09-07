import numpy as np
import matplotlib.pyplot as plt
import my_plotting as mp
from pathlib import Path
from typing import Tuple

from hydro_code_api import HydroCodeAPI as HCA


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

def extract_numbers_from_file(
    hydro_name: str,
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
            use_PT_PL=True,
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

    return np.array([tau / tau[0], e / e[0], pi / h, Pi / h])


if __name__ == "__main__":
    hydro_names = ['ce', 'dnmr', 'mis', 'vah', 'mvah', 'exact']
    colors = mp.get_cmap(10, 'tab10')

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(7, 3 * 7))
    fig.patch.set_facecolor('white')

    for i, hydro_name in enumerate(hydro_names):
        output = extract_numbers_from_file(
            hydro_name=hydro_name,
            path_to_output='../output/mvah_debug',
            mass=0.200 / 0.197,
        )
        for j, y_axis in enumerate([
            r'$\mathcal E / \mathcal E_0$',
            r'$\pi / (\mathcal E + \mathcal P_\mathrm{eq})$',
            r'$\Pi / (\mathcal E + \mathcal P_\mathrm{eq})$'
        ]):
            ax[j].plot(
                output[0],
                output[j + 1],
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
                x_title=r'$\tau / \tau_0$',
                y_title=y_axis
            )
            ax[j].legend(fontsize=20)
            ax[j].set_xscale('log')
    ax[0].set_yscale('log')
    fig.tight_layout()
    fig.savefig('./plots/hydro_compare_1.pdf')
