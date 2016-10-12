#!/usr/bin/env python

import os
import sys
import time
import argparse

import pandas as pd
import numpy as np

from runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from util import *

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import RFECV

if __name__ == '__main__':
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 160)

    parser = argparse.ArgumentParser(description='Get the SVM parameters for all programs.')
    parser.add_argument('--cutoff', dest='cutoff', type=float, default=1.15, help="Cut-off for labelling the runs.")
    parser.add_argument('--uncore', dest='uncore', type=str, help="What uncore counters to include.",
                        default='shared', choices=['all', 'shared', 'exclusive', 'none'])
    parser.add_argument('--config', dest='config', nargs='+', type=str, help="Which configs to include (L3-SMT, L3-SMT-cores, ...).")
    parser.add_argument('data_directory', type=str, help="Data directory root.")
    args = parser.parse_args()

    ## Settings, n_jobs=8:
    MATRIX_FILE = 'matrix_X_uncore_{}.csv'.format(args.uncore)
    CLASSIFIER_CUTOFF = args.cutoff

    results_table = pd.DataFrame()
    runtimes = get_runtime_dataframe(args.data_directory)

    for test in sorted(runtimes['A'].unique()):
        X = pd.DataFrame()
        Y = pd.Series()

        X_test = pd.DataFrame()
        Y_test = pd.Series()

        for config, table in get_runtime_pivot_tables(runtimes):
            if config in args.config:
                for (A, values) in table.iterrows():
                    for (i, normalized_runtime) in enumerate(values):
                        B = table.columns[i]

                        classification = True if normalized_runtime > CLASSIFIER_CUTOFF else False
                        results_path = os.path.join(args.data_directory, config, "{}_vs_{}".format(A, B))
                        matrix_file = os.path.join(results_path, MATRIX_FILE)
                        #print A, B, normalized_runtime, classification

                        if os.path.exists(os.path.join(results_path, 'completed')):
                            if not os.path.exists(matrix_file):
                                print "No matrix file ({}) found, run the scripts/pair/matrix_all.py script first!".format(matrix_file)
                                sys.exit(1)
                            df = pd.read_csv(matrix_file, index_col=False)
                            Y = pd.concat([Y, pd.Series([classification for _ in range(0, df.shape[0])])])
                            X = pd.concat([X, df])
                        else:
                            print "Exclude unfinished directory {}".format(results_path)

        clf = svm.SVC(kernel='linear')

        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)

        selector = RFECV(clf, step=250, cv=1, n_jobs=8)
        selector = selector.fit(X_scaled, Y)

        print selector.support_
        print selector.ranking_
