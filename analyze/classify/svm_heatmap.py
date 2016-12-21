#!/usr/bin/env python3

import os
import sys
import time
import argparse
import re
import subprocess
import math
import logging
from multiprocessing import Pool, TimeoutError, cpu_count

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, font_manager
import matplotlib.cm as cm

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify.svm import get_svm_metrics, CLASSIFIERS, get_argument_parser, make_svm_result_filename, drop_zero_events, make_suffixes
from analyze.classify.svm_topk import get_selected_events
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.classify.generate_matrix import matrix_file_name
from analyze.util import *

import matplotlib
matplotlib.rc('pdf', fonttype=42)
plt.style.use([os.path.join(sys.path[0], '..', 'ethplot.mplstyle')])

def get_matrix_file(args, config, A, B):
    MATRIX_FILE = matrix_file_name(args.core, args.uncore, args.features)
    if B != "Alone":
        results_path = os.path.join(args.data_directory, config, "{}_vs_{}".format(A, B))
    else:
        if args.include_alone:
            results_path = os.path.join(args.data_directory, config, "{}".format(A))
        else:
            return None

    matrix_file_path = os.path.join(results_path, MATRIX_FILE)
    if os.path.exists(os.path.join(results_path, 'completed')):
        if os.path.exists(matrix_file_path):
            return matrix_file_path
        else:
            logging.error("No matrix file ({}) found, run the generate_matrix.py script first!".format(matrix_file_path))
            sys.exit(1)
    else:
        logging.warn("Skipping unfinished directory".format(results_path))
        return None

def cellwise_test_set(args, program_of_interest, program_antagonist, config_of_interest):
    X_test = []
    Y_test = []
    runtime = 0.0

    runtimes = get_runtime_dataframe(args.data_directory)
    for config, table in get_runtime_pivot_tables(runtimes):
        if config in args.config:
            for (A, values) in table.iterrows():
                for (i, normalized_runtime) in enumerate(values):
                    B = table.columns[i]

                    if A == program_of_interest and B == program_antagonist and config == config_of_interest:
                        classification = True if normalized_runtime > args.cutoff else False
                        matrix_file = get_matrix_file(args, config, A, B)
                        if matrix_file == None:
                            continue

                        logging.debug("Adding {} vs. {} in {} to test set".format(A, B, config))
                        df = pd.read_csv(matrix_file, index_col=False)
                        if args.dropzero:
                            drop_zero_events(args, df)

                        X_test.append(df)
                        Y_test.append(pd.Series([classification for _ in range(0, df.shape[0])]))

                        return (pd.concat(X_test), pd.concat(Y_test), normalized_runtime)
                    else:
                        pass

def rowwise_training_set(args, program_of_interest, config_of_interest):
    """
    Includes all cells in the training set except ones that contain data from
    the program of interest (i.e., excludes row and column of the program
    we're evaluating).
    """
    X = []
    Y = []

    runtimes = get_runtime_dataframe(args.data_directory)
    for config, table in get_runtime_pivot_tables(runtimes):
        if config in args.config:
            for (A, values) in table.iterrows():
                for (i, normalized_runtime) in enumerate(values):
                    B = table.columns[i]

                    classification = True if normalized_runtime > args.cutoff else False
                    #if classification == False:
                    # print(A, B, normalized_runtime, classification)
                    matrix_file = get_matrix_file(args, config, A, B)
                    if matrix_file == None:
                        continue

                    if A != program_of_interest and B != program_of_interest:
                        logging.debug("Adding {} vs {} in {} to training set".format(A, B, config))
                        df = pd.read_csv(matrix_file, index_col=False)
                        if args.dropzero:
                            drop_zero_events(args, df)

                        Y.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                        X.append(df)
                    else:
                        pass

    return (pd.concat(X), pd.concat(Y))

def allcells_taining_set(args, program_of_interest, program_antagonist, config_of_interest):
    """
    Includes all cells in the training set except the one we're evaluating.
    """
    X = []
    Y = []

    runtimes = get_runtime_dataframe(args.data_directory)
    for config, table in get_runtime_pivot_tables(runtimes):
        if config in args.config:
            for (A, values) in table.iterrows():
                for (i, normalized_runtime) in enumerate(values):
                    B = table.columns[i]

                    classification = True if normalized_runtime > args.cutoff else False
                    matrix_file = get_matrix_file(args, config, A, B)
                    if matrix_file == None:
                        continue

                    if (A == program_of_interest or A == program_antagonist) and (B == program_antagonist or B == program_of_interest):
                        logging.debug("Skipping {} vs {} in {} from the training set".format(A, B, config))
                    else:
                        logging.debug("Adding {} vs {} in {} to training set".format(A, B, config))
                        df = pd.read_csv(matrix_file, index_col=False)
                        if args.dropzero:
                            drop_zero_events(args, df)

                        Y.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                        X.append(df)

    return (pd.concat(X), pd.concat(Y))


def classify(args, clf, A, columns, config):
    cells = []

    if not args.singlecell:
        X, Y = rowwise_training_set(args, A, config)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        clf.fit(X_scaled, Y)

    for B in columns:
        if B == "Alone":
            continue

        if args.singlecell:
            logging.debug("Training for {} vs {}".format(A, B))
            X, Y = allcells_taining_set(args, A, B, config)
            min_max_scaler = preprocessing.MinMaxScaler()
            X_scaled = min_max_scaler.fit_transform(X)
            clf.fit(X_scaled, Y)

        X_test, Y_test, runtime = cellwise_test_set(args, A, B, config)
        X_test_scaled = min_max_scaler.transform(X_test)
        Y_pred = clf.predict(X_test_scaled)

        pred = get_svm_metrics(args, [A], Y, Y_test, Y_pred)
        pred['NormalizedRuntime'] = runtime
        pred['A'] = A
        pred['B'] = B
        pred['config'] = config
        cells.append(pred)

    print((pd.DataFrame(cells)))
    return pd.DataFrame(cells)

def get_pivot_table(df, idx, value):
    df = df.set_index('config')
    sub_df = df.ix[idx]
    sub_df['Error'] = sub_df['Error'].astype(float)
    sub_df['NormalizedRuntime'] = sub_df['NormalizedRuntime'].astype(float)
    return sub_df.pivot_table(index='A', columns='B', values=value)

def get_config_values(df):
    df = df.set_index('config')
    return df.index.unique()

def generate_heatmaps(args, output_directory, filename, results_table, kconfig):
    for config in get_config_values(results_table):
        error_map = get_pivot_table(results_table, config, 'Error')
        runtimes_map = get_pivot_table(results_table, config, 'NormalizedRuntime')
        plot_filename = filename + "_config_{}".format(config)
        location = os.path.join(output_directory, plot_filename)
        heatmap(args, location, error_map, runtimes_map, config, kconfig, title=not args.paper)

def heatmap(args, location, data, runtimes_map, config, kconfig, title=True):
    fig, ax = plt.subplots()
    label_font = font_manager.FontProperties(family='Supria Sans', size=10)
    ticks_font = font_manager.FontProperties(family='Decima Mono')

    if title:
        alone_suffix, dropzero_suffix, cutoff_suffix = make_suffixes(args)
        title = "Training {}, core {}, uncore {}, features {}, config {}, kernel {}, {}, {}, {}" \
                .format("/".join(sorted(args.config)), args.core, args.uncore, " ".join(sorted(args.features)), \
                        config, kconfig, alone_suffix, cutoff_suffix, dropzero_suffix)
        fig.suptitle(title, fontsize=13, y=1.05)

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

    c = plt.pcolor(data, cmap = cm.Reds, vmin=0.0, vmax=1.0)

    values = data.as_matrix()
    runtimes = runtimes_map.as_matrix()
    for x in range(data.shape[1]):
        for y in range(data.shape[0]):
            if runtimes[x][y] >= (args.cutoff - 0.03) and runtimes[x][y] <= (args.cutoff + 0.03):
                if values[x][y] > 0.50:
                    rect = plt.Rectangle((y,x), 1, 1, color='white', alpha=0.70)
                    ax.add_patch(rect)

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

    if args.paper:
        plt.savefig(location + ".pdf", format='pdf', pad_inches=0.0)

    plt.savefig(location + ".png", format='png')

    plt.clf()
    plt.close()

if __name__ == '__main__':
    parser = get_argument_parser("Compute predicition ability for every cell in the heatmap with all features.")
    parser.add_argument('--singlecell', dest='singlecell', help="Test set is inidvidual-cell, rest is training.", action='store_true', default=False)
    args = parser.parse_args()

    if args.paper:
        output_directory = os.getcwd()
    else:
        output_directory = os.path.join(args.data_directory, "results_svm_heatmap")

    os.makedirs(output_directory, exist_ok=True)

    if args.kernel:
        kernels = [ (args.kernel, CLASSIFIERS[args.kernel]) ]
    else:
        kernels = list(CLASSIFIERS.items())

    for kconfig, clf in kernels:
        print("Trying kernel {}".format(kconfig))

        pool = Pool(processes=cpu_count())
        rows = []
        runtimes = get_runtime_dataframe(args.data_directory)
        for config, table in get_runtime_pivot_tables(runtimes):
            if config in args.config:
                for (A, values) in table.iterrows():
                    res = pool.apply_async(classify, (args, clf, A, table.columns, config))
                    rows.append(res)

        results_table = pd.concat([r.get() for r in rows], ignore_index=True)
        pool.close()
        pool.join()

        if args.singlecell:
            basename = "svm_heatmap_singlecell"
        else:
            basename = "svm_heatmap"

        filename = make_svm_result_filename(basename, args, kconfig)
        results_table.to_csv(os.path.join(output_directory, filename + ".csv"), index=False)
        generate_heatmaps(args, output_directory, filename, results_table, kconfig)
