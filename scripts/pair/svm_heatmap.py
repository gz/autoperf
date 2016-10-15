#!/usr/bin/env python

import os
import sys
import time
import argparse
import re
import subprocess

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, font_manager

from runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from util import *

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

from svm import get_svm_metrics
from svm_topk import get_selected_events

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
                    results_path = os.path.join(args.data_directory, config, "{}_vs_{}".format(A, B))
                    matrix_file = os.path.join(results_path, MATRIX_FILE)
                    #print A, B, normalized_runtime, classification

                    if os.path.exists(os.path.join(results_path, 'completed')):
                        if not os.path.exists(matrix_file):
                            print "No matrix file ({}) found, run the scripts/pair/matrix_all.py script first!".format(matrix_file)
                            sys.exit(1)
                        df = pd.read_csv(matrix_file, index_col=False)

                        if A == program_of_interest and B == program_antagonist and config == config_of_interest:
                            print "Adding {} vs. {} in {} to test set".format(A, B, config)
                            X_test.append(df)
                            Y_test.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                        elif A == program_of_interest or B == program_of_interest:
                            print "Discarding {} vs {} in {}".format(A, B, config)
                        else:
                            print "Adding {} vs {} in {} to training set".format(A, B, config)
                            Y.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                            X.append(df)
                    else:
                        print "Exclude unfinished directory {}".format(results_path)

    return (pd.concat(X), pd.concat(Y), pd.concat(X_test), pd.concat(Y_test))

def mkgroup(cfs_ranking_file):
    ret = subprocess.check_output([AUTOPERF_PATH, "mkgroup", "--input", cfs_ranking_file])
    lines = ret.split(os.linesep)
    assert lines[-1] == ''
    return lines[:-1]

if __name__ == '__main__':
    pd.set_option('display.max_rows', 37)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 200)

    parser = argparse.ArgumentParser(description='Get the SVM parameters when limiting the amount of features.')
    parser.add_argument('--data', dest='data_directory', type=str, help="Data directory root.")

    parser.add_argument('--cutoff', dest='cutoff', type=float, default=1.15, help="Cut-off for labelling the runs.")
    parser.add_argument('--uncore', dest='uncore', type=str, help="What uncore counters to include.",
                        default='shared', choices=['all', 'shared', 'exclusive', 'none'])
    parser.add_argument('--config', dest='config', nargs='+', type=str, help="Which configs to include (L3-SMT, L3-SMT-cores, ...).",
                        default=['L3-SMT', 'L3-SMT-cores'])
    parser.add_argument('--cfs', dest='cfs', type=str, help="Weka file containing reduced, relevant features.")
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="Which programs to use as a test set.")
    args = parser.parse_args()


    if not args.tests:
        runtimes = get_runtime_dataframe(args.data_directory)
        tests = map(lambda x: [x], sorted(runtimes['A'].unique()))
    else:
        tests = [args.tests]

    results_table = pd.DataFrame()

    runtimes = get_runtime_dataframe(args.data_directory)
    for config, table in get_runtime_pivot_tables(runtimes):
        if config in args.config:
            for (A, values) in table.iterrows():
                for (i, normalized_runtime) in enumerate(values):
                    B = table.columns[i]
                    print A, B
                    X, Y, X_test, Y_test = get_training_and_test_set(args, A, B, config)

                    clf = svm.SVC(kernel='linear')
                    min_max_scaler = preprocessing.MinMaxScaler()
                    X_scaled = min_max_scaler.fit_transform(X)
                    X_test_scaled = min_max_scaler.transform(X_test)

                    clf.fit(X_scaled, Y)
                    Y_pred = clf.predict(X_test_scaled)

                    row = get_svm_metrics(args, [A], Y, Y_test, Y_pred)
                    row['A'] = A
                    row['B'] = B
                    row['config'] = config
                    results_table = results_table.append(row, ignore_index=True)
                    print results_table

    results_table.to_csv("svm_heatmap.csv", index=False)
    #results_table = results_table[['Test App', 'Samples', 'Error', 'Precision/Recall', 'F1 score']]
    #print results_table.to_latex(index=False)
