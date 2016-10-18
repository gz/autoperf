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

SVM_KERNELS = {
    #'linear': svm.SVC(kernel='linear'),
    #'linearbalanced': svm.SVC(kernel='linear', class_weight='balanced'),
    #'rbf1': svm.SVC(kernel='rbf', degree=1),
    #'rbf1balanced': svm.SVC(kernel='rbf', degree=1, class_weight='balanced'),
    #'poly1': svm.SVC(kernel='poly', degree=1),
    'poly2': svm.SVC(kernel='poly', degree=2),
    #'poly1balanced': svm.SVC(kernel='poly', degree=1, class_weight='balanced'),
    #'poly2balanced': svm.SVC(kernel='poly', degree=2, class_weight='balanced'),
}

def get_argument_parser(desc):
    pd.set_option('display.max_rows', 37)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 200)

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data', dest='data_directory', type=str, help="Data directory root.")
    parser.add_argument('--cutoff', dest='cutoff', type=float, default=1.15, help="Cut-off for labelling the runs.")
    parser.add_argument('--uncore', dest='uncore', type=str, help="What uncore counters to include.",
                        default='shared', choices=['all', 'shared', 'exclusive', 'none'])
    parser.add_argument('--config', dest='config', nargs='+', type=str, help="Which configs to include (L3-SMT, L3-SMT-cores, ...).",
                        default=['L3-SMT'])
    parser.add_argument('--alone', dest='include_alone', action='store_true',
                        default=False, help="Include alone runs.")

    return parser

def row_training_and_test_set(args, tests):
    MATRIX_FILE = 'matrix_X_uncore_{}.csv'.format(args.uncore)

    X = []
    Y = []
    Y_weights = []

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
                        if not args.include_alone:
                            #print "Skipping the samples with {} alone".format(A)
                            continue
                        results_path = os.path.join(args.data_directory, config, "{}".format(A))
                    else:
                        results_path = os.path.join(args.data_directory, config, "{}_vs_{}".format(A, B))
                    matrix_file = os.path.join(results_path, MATRIX_FILE)

                    if os.path.exists(os.path.join(results_path, 'completed')):
                        if not os.path.exists(matrix_file):
                            print "No matrix file ({}) found, run the scripts/pair/matrix_all.py script first!".format(matrix_file)
                            sys.exit(1)
                        df = pd.read_csv(matrix_file, index_col=False)

                        if A in tests:
                            #print "Adding {} vs {} to test set".format(A, B), classification
                            Y_test.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                            X_test.append(df)
                        elif B in tests:
                            #print "Discarding {} vs {}".format(A, B), classification
                            pass
                        else:
                            Y.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                            if (A == "NBODY" and (B == "CNEAL" or B == "HD1400" or B == "PR700" or B == "AA700")) \
                            or (B == "NBODY" and (A == "HD1400" or A == "CNEAL" or A == "BSCHOL")) \
                            or (A == "TC1400" and B == "BSCHOL") \
                            or (A == "HD1400" and (B == "SCLUS" or B == "SWAPT")):
                                print "Give more weights to {} vs {}".format(A, B)
                                Y_weights.append(pd.Series([6 for _ in range(0, df.shape[0])]))
                            else:
                                Y_weights.append(pd.Series([1 for _ in range(0, df.shape[0])]))
                                #print "Increase weights for BSCHOL!"

                            X.append(df)
                    else:
                        print "Exclude unfinished directory {}".format(results_path)

    return (pd.concat(X), pd.concat(Y), pd.concat(Y_weights), pd.concat(X_test), pd.concat(Y_test))

def get_svm_metrics(args, test, Y, Y_test, Y_pred):
    row = {}
    row['Training Configs'] = ' '.join(args.config)
    row['Tested Application'] = ' '.join(test)

    row['Samples Training Total'] = "{}".format(len(Y))
    row['Samples Test Total'] = "{}".format(len(Y_test))

    row['Samples Training 0'] = len(Y[Y == 0])
    row['Samples Training 1'] = len(Y[Y == 1])
    row['Samples Test 0'] = len(Y_test[Y_test == 0])
    row['Samples Test 1'] = len(Y_test[Y_test == 1])

    row['Accuracy'] = "%.2f" % metrics.accuracy_score(Y_test, Y_pred)
    row['Error'] = "%.2f" % (1.0 - metrics.accuracy_score(Y_test, Y_pred))
    row['Precision'] = "%.2f" % metrics.precision_score(Y_test, Y_pred)
    row['Recall'] = "%.2f" % metrics.recall_score(Y_test, Y_pred)
    row['F1 score'] = "%.2f" % metrics.f1_score(Y_test, Y_pred)

    return row

def make_result_filename(prefix, args, kconfig):
    alone_suffix = "alone" if args.include_alone else "paironly"
    cutoff_suffix = "{}".format(args.cutoff*100)
    filename = prefix + "_training_{}_uncore_{}_{}_{}_{}" \
               .format("_".join(args.config), args.uncore, kconfig, alone_suffix, cutoff_suffix)
    return filename

if __name__ == '__main__':
    parser = get_argument_parser('Get the SVM parameters for a row in the heatmap.')
    parser.add_argument('--weka', dest='weka', action='store_true', default=False, help='Save files for Weka')
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="List or programs to include for the test set.")
    args = parser.parse_args()

    if not args.tests:
        runtimes = get_runtime_dataframe(args.data_directory)
        tests = map(lambda x: [x], sorted(runtimes['A'].unique())) # None here means we save the whole matrix as X (no training set)
    else:
        tests = [args.tests] # Pass the tests as a single set

    for kconfig, clf in SVM_KERNELS.iteritems():
        print "Trying kernel", kconfig
        results_table = pd.DataFrame()

        for test in tests:
            X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args, test)
            min_max_scaler = preprocessing.MinMaxScaler()
            X_scaled = min_max_scaler.fit_transform(X)

            if test != [None]:
                X_test_scaled = min_max_scaler.transform(X_test)

                clf.fit(X_scaled, Y, sample_weight=Y_weights)
                Y_pred = clf.predict(X_test_scaled)

                row = get_svm_metrics(args, test, Y, Y_test, Y_pred)
                results_table = results_table.append(row, ignore_index=True)
                print results_table

        filename = make_result_filename("svm_results", args, kconfig)
        results_table.to_csv(filename + ".csv", index=False)

    if args.weka:
        for test in tests:
            # TODO: Weka has a bug when the 2nd class appears late in the vector it will think this
            # file has only one class and complain. THe solutionis to make sure both class label appear
            # directly for example as first and 2nd row XD
            X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args, test)

            X['Y'] = Y
            X_test['Y'] = Y_test

            X['Y'] = X['Y'].map(lambda x: 'Y' if x else 'N')
            X_test['Y'] = X_test['Y'].map(lambda x: 'Y' if x else 'N')

            training_file_name = "unset"
            if test == [None]:
                training_file_name = 'svm_complete_{}_uncore_{}.csv'.format('_'.join(args.config), args.uncore)
            else:
                training_file_name = 'svm_training_without_{}_{}_uncore_{}.csv'.format('_'.join(test), '_'.join(args.config), args.uncore)

            X.to_csv(os.path.join(args.data_directory, training_file_name), index=False)

            if test != [None]:
                test_file_name = 'svm_test_{}_{}_uncore_{}.csv'.format('_'.join(test), '_'.join(args.config), args.uncore)
                X_test.to_csv(os.path.join(args.data_directory, test_file_name), index=False)
