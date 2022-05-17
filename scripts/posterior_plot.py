#!/bin/python3
import pickle

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_plotting import costumize_axis, smooth_histogram

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)


def analyze_saved_runs(number_of_runs: int) -> None:
    # range for hist binning
    hist_range = (0.2, 0.5)
    # data frames to be used by seaborn and draw posteriors
    dfs = pd.DataFrame(columns=[r'$\mathcal C$', 'hydro'])
    df_special = pd.DataFrame(columns=[r'$\mathcal C$', 'weight', 'hydro'])
    all_counts = dict((key, []) for key in ['ce', 'dnmr', 'vah', 'mvah'])
    x = np.random.randint(number_of_runs)
    for i in range(number_of_runs):
        with open(f'mass_MCMC_run_{i}.pkl', 'rb') as f:
            mcmc_chains = pickle.load(f)
            for key in mcmc_chains.keys():
                data = mcmc_chains[key][0].reshape(-1, 1)
                counts, bins, p = plt.hist(data.flatten(),
                                           bins=200,
                                           range=hist_range)
                all_counts[key].append(counts / np.sum(counts * np.diff(bins)))
                df = pd.DataFrame({r'$\mathcal C$': data[:, 0]})
                df['hydro'] = key
                dfs = pd.concat([dfs, df], ignore_index=True)

                # need to create a histogram, and keep track of the bins
                # then plot the normalizes hist for the comparison
                if x == i:
                    counts_special, bins_special, p_special = plt.hist(
                        data,
                        bins=200,
                        range=hist_range)
                    df_special = pd.concat([
                        df_special,
                        pd.DataFrame(
                            {r'$\mathcal C$': bins_special[:-1],
                             'weight': counts_special
                             / np.sum(counts_special * np.diff(bins)),
                             'hydro': key})],
                        ignore_index=True)

    df_spread = pd.DataFrame(columns=[r'$\mathcal C$',
                                      r'$-\sigma$',
                                      r'$\mu$',
                                      r'$+\sigma$',
                                      'hydro'])

    col_names = [r'$-\sigma$', r'$\mu$', r'$+\sigma$']
    for key in all_counts.keys():
        data = np.array(all_counts[key])
        high_low = np.quantile(data, [0.16, 0.50, 0.84], axis=0)
        df = pd.DataFrame({
            r'$\mathcal C$': bins[:-1],
            r'$-\sigma$': high_low[0],
            r'$\mu$': high_low[1],
            r'$+\sigma$': high_low[2],
            'hydro': key
        })
        df_spread = pd.concat([df_spread, df], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    x_axis_col = r'$\mathcal C$'
    for name in col_names:
        if name == r'$\mu$':
            sns.kdeplot(
                data=df_spread,
                x=x_axis_col,
                weights=df_spread[name],
                hue='hydro',
                ax=ax)
        else:
            sns.kdeplot(
                data=df_spread,
                x=x_axis_col,
                weights=df_spread[name],
                hue='hydro',
                linestyle='dashed',
                ax=ax)
    lines_index = [[0, 8],      # ce 1st std lines plotted
                   [1, 9],      # dmnr
                   [2, 10],     # vah
                   [3, 11]]     # mvah
    plotted_lines = ax.get_lines()
    cmap = plt.get_cmap('tab10', 10)
    for k, pairs in enumerate(lines_index):
        ax.fill_between(x=plotted_lines[pairs[0]].get_xdata(),
                        y1=plotted_lines[pairs[0]].get_ydata(),
                        y2=plotted_lines[pairs[1]].get_ydata(),
                        color=cmap(3-k),    # Lines seem to be fed like a queue
                        alpha=0.4)
    ax.set_xlim(*hist_range)
    fig.tight_layout()
    fig.savefig('upper-lower.pdf')
    del fig, ax

    g2 = sns.pairplot(data=dfs,
                      corner=True,
                      diag_kind='kde',
                      kind='hist',
                      hue='hydro')
    g2.map_lower(sns.kdeplot, levels=4)
    g2.tight_layout()
    g2.savefig('full-posterior.pdf')
    del g2

    # compare one to all
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('white')
    sns.kdeplot(data=dfs,
                x=x_axis_col,
                hue='hydro',
                ax=ax)
    sns.kdeplot(data=df_special,
                x=x_axis_col,
                weights=df_special['weight'],
                hue='hydro',
                linestyle='dashed',
                alpha=0.5,
                ax=ax)
    ax.set_xlim(*hist_range)
    fig.tight_layout()
    fig.savefig('compare-one-to-many.pdf')
    del fig, ax


def analyze_saved_runs_hist(number_of_runs: int) -> None:
    # range for hist binning
    hist_range = (0.2, 0.5)
    # data frames to be used by seaborn and draw posteriors
    dfs = pd.DataFrame(columns=[r'$\mathcal C$', 'hydro'])
    df_special = pd.DataFrame(columns=[r'$\mathcal C$', 'weight', 'hydro'])
    all_counts = dict((key, []) for key in ['ce', 'dnmr', 'vah', 'mvah'])
    x = np.random.randint(number_of_runs)
    for i in range(number_of_runs):
        with open(f'mass_MCMC_run_{i}.pkl', 'rb') as f:
            mcmc_chains = pickle.load(f)
            for key in mcmc_chains.keys():
                data = mcmc_chains[key][0].reshape(-1, 1)
                counts, bins, p = plt.hist(data.flatten(),
                                           bins=200,
                                           range=hist_range)
                all_counts[key].append(smooth_histogram(
                    counts=counts / np.sum(counts * np.diff(bins)),
                    window_size=int(np.sqrt(counts.size))))
                df = pd.DataFrame({r'$\mathcal C$': data[:, 0]})
                df['hydro'] = key
                dfs = pd.concat([dfs, df], ignore_index=True)

                # need to create a histogram, and keep track of the bins
                # then plot the normalizes hist for the comparison
                if x == i:
                    counts_special, bins_special, p_special = plt.hist(
                        data,
                        bins=200,
                        range=hist_range)
                    df_special = pd.concat([
                        df_special,
                        pd.DataFrame(
                            {r'$\mathcal C$': bins_special[:-1],
                             'weight': counts_special
                             / np.sum(counts_special * np.diff(bins)),
                             'hydro': key})],
                        ignore_index=True)

    df_spread = pd.DataFrame(columns=[r'$\mathcal C$',
                                      r'$-\sigma$',
                                      r'$\mu$',
                                      r'$+\sigma$',
                                      'hydro'])

    col_names = [r'$-\sigma$', r'$\mu$', r'$+\sigma$']
    for key in all_counts.keys():
        data = np.array(all_counts[key])
        high_low = np.quantile(data, [0.16, 0.50, 0.84], axis=0)
        df = pd.DataFrame({
            r'$\mathcal C$': bins[:-1],
            r'$-\sigma$': np.array(high_low[0], dtype=float),
            r'$\mu$': np.array(high_low[1], dtype=float),
            r'$+\sigma$': np.array(high_low[2], dtype=float),
            'hydro': key
        })
        df_spread = pd.concat([df_spread, df], ignore_index=True)

    cmap = plt.get_cmap('tab10', 10)
    fig, ax = plt.subplots(figsize=(7, 7))
    costumize_axis(ax, r'$\mathcal C$', 'Density')
    x_axis_col = r'$\mathcal C$'
    for i, hydro in enumerate(all_counts.keys()):
        ax.plot(df_spread.loc[df_spread['hydro'] == hydro][x_axis_col],
                smooth_histogram(
                    counts=df_spread.
                    loc[df_spread['hydro'] == hydro][col_names[1]].to_numpy(),
                    window_size=int(np.sqrt(
                        df_spread.
                        loc[df_spread['hydro'] == hydro][col_names[1]].size
                    ))),
                color=cmap(i),
                lw=2,
                label=hydro)

        x = np.array(df_spread.loc[df_spread['hydro'] == hydro][x_axis_col]
                     .to_numpy(), dtype=np.float32)
        y1 = np.array(df_spread.loc[df_spread['hydro'] == hydro][col_names[0]]
                      .to_numpy(), dtype=np.float32)
        y2 = np.array(df_spread.loc[df_spread['hydro'] == hydro][col_names[2]]
                      .to_numpy(), dtype=np.float32)

        x = smooth_histogram(counts=x, window_size=int(np.sqrt(x.size)))
        y1 = smooth_histogram(counts=y1,
                              window_size=int(np.sqrt(y1.size)))
        y2 = smooth_histogram(counts=y2,
                              window_size=int(np.sqrt(y2.size)))
        ax.fill_between(x=x,
                        y1=y1,
                        y2=y2,
                        color=cmap(i),
                        alpha=0.4)
    ax.set_xlim(*hist_range)
    ax.legend(fontsize=18)
    fig.tight_layout()
    fig.savefig('upper-lower_hist.pdf')
    del fig, ax

    fig2, ax2 = plt.subplots(figsize=(7, 7))
    fig2.patch.set_facecolor('white')
    costumize_axis(ax2, x_axis_col, 'Density')
    for i, hydro in enumerate(all_counts.keys()):
        counts, bins = np.histogram(
            a=dfs.loc[dfs['hydro'] == hydro][x_axis_col],
            bins=200,
            normed=True)
        ax2.hist(
            x=bins[:-1],
            bins=bins,
            weights=smooth_histogram(counts=counts,
                                     window_size=int(np.sqrt(counts.size))),
            color=cmap(i),
            histtype=u'step',
            lw=2
        )
    ax2.set_xlim(*hist_range)
    fig2.tight_layout()
    fig2.savefig('full-posterior_hist.pdf')
    del fig2, ax2

    # compare one to all
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    fig3.patch.set_facecolor('white')
    costumize_axis(ax3, x_axis_col, 'Density')
    for i, hydro in enumerate(all_counts.keys()):
        counts, bins = np.histogram(
            a=dfs.loc[dfs['hydro'] == hydro][x_axis_col],
            bins=200,
            normed=True)
        ax3.hist(
            x=bins[:-1],
            bins=bins,
            weights=smooth_histogram(counts=counts,
                                     window_size=int(np.sqrt(counts.size))),
            color=cmap(i),
            histtype=u'step',
            lw=2)

        counts = np.array(
            df_special.loc[df_special['hydro'] == hydro]['weight']
            .to_numpy(),
            dtype=float)
        ax3.hist(
            x=bins_special[:-1],
            bins=bins_special,
            weights=  # df_special.loc[df_special['hydro'] == hydro]['weight'],
            smooth_histogram(
                counts=counts,
                window_size=int(np.sqrt(counts.size))),
            color=cmap(i),
            histtype=u'step',
            ls='dashed',
            lw=2,
            alpha=0.5)
    ax3.set_xlim(*hist_range)
    fig3.tight_layout()
    fig3.savefig('compare-one-to-many_hist.pdf')
    del fig3, ax3


if __name__ == "__main__":
    analyze_saved_runs(number_of_runs=38)
    analyze_saved_runs_hist(number_of_runs=38)
