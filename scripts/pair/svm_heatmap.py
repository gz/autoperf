#!/usr/bin/env python

import os
import sys
import time
import argparse
import re
import subprocess
from multiprocessing import Pool, TimeoutError

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, font_manager
import matplotlib.cm as cm

from runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from util import *

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

from svm import get_svm_metrics, SVM_KERNELS, get_argument_parser
from svm_topk import get_selected_events

plt.style.use([os.path.join(sys.path[0], '..', 'ethplot.mplstyle')])
AUTOPERF_PATH = os.path.join(sys.path[0], "..", "..", "target", "release", "autoperf")

def get_training_and_test_set(args, program_of_interest, program_antagonist, config_of_interest):
    MATRIX_FILE = 'matrix_X_uncore_{}.csv'.format(args.uncore)

    X = []
    Y = []

    X_test = []
    Y_test = []

    runtimes = get_runtime_dataframe(args.data_directory)
    for config, table in get_runtime_pivot_tables(runtimes):
        if config in args.config:
            for (A, values) in table.iterrows():
                for (i, normalized_runtime) in enumerate(values):
                    B = table.columns[i]

                    classification = True if normalized_runtime > args.cutoff else False
                    if B == "Alone":
                        if args.include_alone:
                            results_path = os.path.join(args.data_directory, config, "{}".format(A))
                            #print "Include", results_path
                        else:
                            continue
                    else:
                        results_path = os.path.join(args.data_directory, config, "{}_vs_{}".format(A, B))
                    matrix_file = os.path.join(results_path, MATRIX_FILE)
                    #print A, B, normalized_runtime, classification

                    if os.path.exists(os.path.join(results_path, 'completed')):
                        if not os.path.exists(matrix_file):
                            print "No matrix file ({}) found, run the scripts/pair/matrix_all.py script first!".format(matrix_file)
                            sys.exit(1)
                        df = pd.read_csv(matrix_file, index_col=False)

                        if A == program_of_interest and B == program_antagonist and config == config_of_interest:
                            #print "Adding {} vs. {} in {} to test set".format(A, B, config)
                            X_test.append(df)
                            Y_test.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                        elif A == program_of_interest or B == program_of_interest:
                            #print "Discarding {} vs {} in {}".format(A, B, config)
                            pass
                        else:
                            #print "Adding {} vs {} in {} to training set".format(A, B, config)
                            Y.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                            X.append(df)
                    else:
                        print "Exclude unfinished directory {}".format(results_path)

    return (pd.concat(X), pd.concat(Y), pd.concat(X_test), pd.concat(Y_test))

def classify(args, clf, A, B, config):
    X, Y, X_test, Y_test = get_training_and_test_set(args, A, B, config)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    X_test_scaled = min_max_scaler.transform(X_test)

    clf.fit(X_scaled, Y)
    Y_pred = clf.predict(X_test_scaled)

    row = get_svm_metrics(args, [A], Y, Y_test, Y_pred)
    row['A'] = A
    row['B'] = B
    row['config'] = config
    print pd.DataFrame([row])
    return pd.DataFrame([row])

def get_pivot_tables(df):
    df = df.set_index('config')

    tables = []
    for idx in df.index.unique():
        sub_df = df.ix[idx]
        sub_df['Error'] = sub_df['Error'].astype(float)
        pivot_table = sub_df.pivot_table(index='A', columns='B', values='Error')
        tables.append( (idx, pivot_table) )

    return tables

def heatmap(location, data, title):
    fig, ax = plt.subplots()
    label_font = font_manager.FontProperties(family='Supria Sans', size=10)
    ticks_font = font_manager.FontProperties(family='Decima Mono')

    if title:
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
    parser = get_argument_parser("Compute predicition ability for every cell in the heatmap with all features.")
    args = parser.parse_args()

    for kconfig, clf in SVM_KERNELS.iteritems():
        print "Trying kernel", kconfig

        pool = Pool(processes=6)
        rows = []
        runtimes = get_runtime_dataframe(args.data_directory)
        for config, table in get_runtime_pivot_tables(runtimes):
            if config in args.config:
                for (A, values) in table.iterrows():
                    for (i, normalized_runtime) in enumerate(values):
                        B = table.columns[i]
                        if B == "Alone":
                            continue
                        res = pool.apply_async(classify, (args, clf, A, B, config))
                        rows.append(res)

        results_table = pd.concat(map(lambda r: r.get(), rows), ignore_index=True)
        pool.close()
        pool.join()

        alone_suffix = "alone" if args.alone else "paironly"
        cutoff_suffix = "{}".format(args.cutoff*100)

        filename = "svm_heatmap_training_{}_uncore_{}_{}_{}_{}" \
                   .format("_".join(args.config), args.uncore, kconfig, alone_suffix, cutoff_suffix)
        results_table.to_csv(filename + ".csv", index=False)

        for (config, pivot_table) in get_pivot_tables(results_table):
            plot_filename = filename + "_config_{}".format(config)
            title = "Training {}, uncore {}, config {}, kernel {}, {}, {}" \
                    .format("/".join(args.config), args.uncore, config, kconfig, alone_suffix, cutoff_suffix)
            heatmap(filename, pivot_table, title)
