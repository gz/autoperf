#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gathers the runtimes from all pairwise runs and makes a heatmap
out of it.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt, font_manager
from matplotlib.colors import Normalize, LinearSegmentedColormap

colors = LinearSegmentedColormap.from_list('seismic', ["#ca0020", "#2ca25f"])

def get_runtime_dataframe(data_directory):
    runtimes_file = os.path.join(data_directory, 'runtimes.csv')
    if os.path.exists(runtimes_file):
        return pd.read_csv(os.path.join(runtimes_file))
    else:
        print "{} does not exist yet, make sure you call scripts/pair/runtimes.py!".format(runtimes_file)

def compute_runtime_dataframe(data_directory):
    """
    Walks through all subdirectories in the results directory.
    Then finds all that have a 'completed' file and gathers all
    the runtimes from every run in perf.csv.

    - Ignores any results that have stderr output in them.
    - Prints a warning in case stdev is more than 1sec.
    - TODO: Should probably have return codes in perf.csv and check them as well!

    Finally returns a dataframe that looks a bit like this:
    =============================================================
              A       B        config          mean           std
    0     AA700   Alone        L1-SMT  31966.062500    240.786896
    1     PR700   PR700        L1-SMT  88666.857143    262.895069
    2     PF200   Alone        L1-SMT  20426.687500     90.623767
    ...
    """

    PARSEC_TIME_PATTERN = "[HOOKS] Total time spent in ROI:"
    GREEN_MARL_TIME_PATTERN = "running time="

    row_list = []
    for root, dirs, files in os.walk(data_directory):
        if os.path.exists(os.path.join(root, 'completed')):
            row = {}

            perf_csv = os.path.join(root, 'perf.csv')
            rest, programs = os.path.split(root)
            _, config = os.path.split(rest)
            programs = programs.split("_vs_")

            row['A'] = programs[0]
            row['B'] = "Alone" if len(programs) == 1 else programs[1]
            row['config'] = config

            df = pd.read_csv(perf_csv, skipinitialspace=True)
            # Because I was stupid and called stderr stdout originally
            stderr_key = 'stdin' if 'stdin' in df.columns else 'stderr'
            if not df[stderr_key].isnull().values.all() and not row['A'] in ['SWAPT', 'SCLUS']:
                print root, "has errors. Don't put it in the matrix!"
                continue

            runtimes = []
            for out in df['stdout']:
                lines = out.split('\n')
                time_ms = None
                for line in lines:
                    if line.startswith(PARSEC_TIME_PATTERN):
                        time_s_str = line.split(PARSEC_TIME_PATTERN)[1].strip()
                        time_sec = float(time_s_str[:-1])
                        time_ms = int(time_sec * 1000)
                    if line.startswith(GREEN_MARL_TIME_PATTERN):
                        # Drop the s at the end
                        time_ms_str = line.split(GREEN_MARL_TIME_PATTERN)[1].strip()
                        time_ms = int(time_ms_str.split(".")[0])

                assert time_ms is not None
                runtimes.append(time_ms)

            rseries = pd.Series(runtimes)
            name = "{}_vs_{}".format(row['A'], row['B'])
            violin("violin_{}_{}".format(config, name), name, rseries)

            row['A mean'] = rseries.mean()
            row['A std'] = rseries.std()
            if row['A std'] > 1000:
                print "{}: High standard deviation {} with mean {}.".format(root, row['A std'], row['A mean'])

            row_list.append(row)
        else:
            print "Exclude unfinished directory {}".format(root)


    return pd.DataFrame(row_list)


def get_runtime_pivot_tables(df):
    """
    Takes as input the dataframe provided by 'compute_runtime_dataframe'.

    Returns as output a list of tuples (config, pivot table) for every configuration
    that shows the runtimes of the program of every row running together with
    the program of the column normalized (compared to running alone).
    """
    df = df.set_index('config')

    tables = []
    for idx in df.index.unique():
        sub_df = df.ix[idx]

        normalize_by = {}
        for (key, row) in sub_df.iterrows():
            if row['B'] == "Alone" or pd.isnull(row['B']):
                normalize_by[row['A']] = row['A mean']

        def add_normalized(x):
            x['A alone'] = normalize_by[x['A']]
            x['A mean normalized'] = x['A mean'] / normalize_by[x['A']]
            return x

        sub_df = sub_df.apply(add_normalized, axis=1, reduce=False)
        pivot_table = sub_df.pivot_table(index='A', columns='B', values='A mean normalized')
        tables.append( (idx, pivot_table) )

    return tables

def violin(location, name, data):
    fig, ax = plt.subplots()

    ax.violinplot(data, points=20, widths=0.1, showmeans=True, showextrema=True, showmedians=True)
    ax.set_title(name)

    plt.savefig(location + ".png", format='png')
    #plt.savefig(location + ".pdf", format='pdf', pad_inches=0.0)
    plt.clf()
    plt.close()


def heatmap(location, data):
    del data['Alone'] # We don't need this in the heatmaps...
    fig, ax = plt.subplots()
    label_font = font_manager.FontProperties(family='Supria Sans', size=10)
    ticks_font = font_manager.FontProperties(family='Decima Mono')

    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.index)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    ax.tick_params(pad=11)

    plt.setp(ax.get_xticklabels(), fontproperties=label_font)
    plt.setp(ax.get_yticklabels(), fontproperties=label_font)

    c = plt.pcolor(data, cmap = cm.Greys, vmin=1.0, vmax=1.5)

    values = data.as_matrix()
    for x in range(data.shape[1]):
        for y in range(data.shape[0]):
            color = 'white' if values[y][x] > 1.4 else 'black'
            plt.text(x + 0.5, y + 0.5, '%.2f' % values[y][x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color=color,
                     fontproperties=ticks_font)

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    plt.savefig(location + ".png", format='png')
    plt.savefig(location + ".pdf", format='pdf', pad_inches=0.0)
    plt.clf()
    plt.close()

if __name__ == '__main__':
    plt.style.use([os.path.join(os.path.realpath(__file__), '..', 'ethplot.mplstyle')])
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 160)

    parser = argparse.ArgumentParser(description='Gather all runtimes and plot heatmaps.')
    parser.add_argument('data_directory', type=str, help="Data directory root.")
    args = parser.parse_args()

    df = compute_runtime_dataframe(args.data_directory)
    df = df[['config', 'A', 'B', 'A mean', 'A std']]
    df.to_csv(os.path.join(args.data_directory, "runtimes.csv"), index=False)

    for config, pivot_table in get_runtime_pivot_tables(df):
        heatmap(os.path.join(args.data_directory, config), pivot_table)
