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
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 5)
    pd.set_option('display.width', 160)

    parser = argparse.ArgumentParser(description='Get the SVM accuracy for all programs.')
    parser.add_argument('--config', dest='config', nargs='+', type=str, help="Which configs to include (L3-SMT, L3-SMT-cores, ...).")
    parser.add_argument('--uncore', dest='uncore', type=str, help="What uncore counters to include.", default='shared')
    parser.add_argument('--test', dest='test', nargs='+', type=str, help="Which programs to use as the test set.")
    parser.add_argument('data_directory', type=str, help="Data directory root.")
    args = parser.parse_args()

    ## Settings:
    MATRIX_FILE = 'matrix_X_uncore_{}.csv'.format(args.uncore)
    CLASSIFIER_CUTOFF = 1.15

    X_test = pd.DataFrame()
    Y_test = pd.Series()

    X = pd.DataFrame()
    Y = pd.Series()

    runtimes = get_runtime_dataframe(args.data_directory)

    for config, table in get_runtime_pivot_tables(runtimes):
        if config in args.config:
            for (A, values) in table.iterrows():
                for (i, normalized_runtime) in enumerate(values):
                    B = table.columns[i]

                    classification = 'Y' if normalized_runtime > CLASSIFIER_CUTOFF else 'N'
                    results_path = os.path.join(args.data_directory, config, "{}_vs_{}".format(A, B))
                    matrix_file = os.path.join(results_path, MATRIX_FILE)

                    if os.path.exists(os.path.join(results_path, 'completed')):
                        if not os.path.exists(matrix_file):
                            print "No matrix file ({}) found, run the scripts/pair/matrix_all.py script first!".format(matrix_file)
                            sys.exit(1)
                        df = pd.read_csv(matrix_file, index_col=False)

                        if A in args.test or B in args.test:
                            #print "Adding {} vs {} to test set".format(A, B)
                            Y_test = pd.concat([Y_test, pd.Series([classification for _ in range(0, df.shape[0])])])
                            X_test = pd.concat([X_test, df])
                        else:
                            Y = pd.concat([Y, pd.Series([classification for _ in range(0, df.shape[0])])])
                            X = pd.concat([X, df])
                    else:
                        print "Exclude unfinished directory {}".format(results_path)


    training_file_name = 'svm_training_without_{}_{}_uncore_{}.csv'.format('_'.join(args.test), '_'.join(args.config), args.uncore)
    X['Y'] = Y
    #print X
    X.to_csv(os.path.join(args.data_directory, training_file_name), index=False)

    test_file_name = 'svm_test_{}_{}_uncore_{}.csv'.format('_'.join(args.test), '_'.join(args.config), args.uncore)
    X_test['Y'] = Y_test
    #print X_test
    X_test.to_csv(os.path.join(args.data_directory, test_file_name), index=False)

    X['Y'] = X['Y'].map(lambda x: 1 if x == 'Y' else 0)
    X_test['Y'] = X_test['Y'].map(lambda x: 1 if x == 'Y' else 0)

    clf = svm.SVC(kernel='poly', degree=1, coef0=250007)
    clf.fit(X.as_matrix(), Y.as_matrix())
    print "Test {}".format(args.test), clf.score(X_test, Y_test)

    #model = svm.SVC()
    #print cross_val_score(model, X_test, Y_test, scoring='accuracy')
