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
from analyze.classify.svm_heatmap import get_matrix_file, classify, generate_heatmaps
from analyze.classify.svm_topk import get_selected_events
from analyze.classify.runtimes import get_runtime_dataframe, get_max_runtime_pivot_tables
from analyze.classify.generate_matrix import matrix_file_name
from analyze.util import *

import matplotlib
matplotlib.rc('pdf', fonttype=42)
plt.style.use([os.path.join(sys.path[0], '..', 'ethplot.mplstyle')])

def cellwise_test_set(args, program_of_interest, program_antagonist, config_of_interest):
    X_test = []
    Y_test = []
    runtime = 0.0

    runtimes = get_runtime_dataframe(args.data_directory)
    for config, table in get_max_runtime_pivot_tables(runtimes):
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
    for config, table in get_max_runtime_pivot_tables(runtimes):
        if config in args.config:
            for (A, values) in table.iterrows():
                for (i, normalized_runtime) in enumerate(values):
                    B = table.columns[i]

                    classification = True if normalized_runtime > args.cutoff else False
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


if __name__ == '__main__':
    parser = get_argument_parser("Compute predicition ability for pairs in the heatmap (with all features).")
    parser.add_argument('--singlecell', dest='singlecell', help="Test set is inidvidual-cell, rest is training.", action='store_true', default=False)
    args = parser.parse_args()

    if args.paper:
        output_directory = os.getcwd()
    else:
        output_directory = os.path.join(args.data_directory, "results_svm_heatmap_socket")

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
        for config, table in get_max_runtime_pivot_tables(runtimes):
            if config in args.config:
                for (A, values) in table.iterrows():
                    res = pool.apply_async(classify, (args, clf, A, table.columns, config))
                    rows.append(res)

        results_table = pd.concat([r.get() for r in rows], ignore_index=True)
        pool.close()
        pool.join()

        basename = "svm_socket_heatmap"
        filename = make_svm_result_filename(basename, args, kconfig)
        results_table.to_csv(os.path.join(output_directory, filename + ".csv"), index=False)
        generate_heatmaps(args, output_directory, filename, results_table, kconfig)
