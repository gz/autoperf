#!/usr/bin/env python3

import os
import sys
import time
import argparse
import math
import logging

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import neural_network
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify import get_argument_parser
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.classify.generate_matrix import matrix_file_name
from analyze.util import *


SMT_SPEEDUP = {
    'PR700': 1.0,
    'AA700': 1.33,
    'HD1400': 1.5,
    'TC1400': 1.53,
    'NBODY': 1.05,
    'BSCHOL': 1.74,
    'CNEAL': 1.48,
    'FERR': 1.33,
    'SCLUS': 1.07,
    'SWAPT': 1.07
}

def drop_zero_events(args, df):
    from analyze.classify.find_all_zero import zero_features
    to_drop = zero_features(args, overwrite=False)
    df.drop(to_drop['EVENT_NAME'], axis=1, inplace=True)

def row_training_and_test_set(args, tests):
    assert(len(tests) == 1)
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

                    classification = True if SMT_SPEEDUP[A] > args.cutoff else False
                    if B != "Alone":
                        logging.debug("Skipping the samples that are not alone")
                        continue

                    results_path = os.path.join(args.data_directory, config, "{}".format(A))
                    matrix_file_path = os.path.join(results_path, matrix_file_name(args.core, args.uncore, args.features))

                    if os.path.exists(os.path.join(results_path, 'completed')):
                        if not os.path.exists(matrix_file_path):
                            logging.error("No matrix file ({}) found, run the generate_matrix.py script first!".format(matrix_file_path))
                            sys.exit(1)

                        df = pd.read_csv(matrix_file_path, index_col=False)
                        if args.dropzero:
                            #logging.debug("Dropping zero")
                            drop_zero_events(args, df)
                        if A in tests:
                            logging.debug("Adding {} vs {} to test set class={} file={}".format(A, B, classification, matrix_file_path))
                            Y_test.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                            X_test.append(df)
                        elif B in tests:
                            logging.debug("Discarding {} vs {} class={} file={}".format(A, B, classification, matrix_file_path))
                            pass
                        else:
                            logging.debug("Adding {} vs {} to training set class={} file={}".format(A, B, classification, matrix_file_path))
                            Y.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                            Y_weights.append(pd.Series([1 for _ in range(0, df.shape[0])]))
                            X.append(df)
                    else:
                        print(("Exclude unfinished directory {}".format(results_path)))

    return (pd.concat(X), pd.concat(Y), pd.concat(Y_weights),
            pd.concat(X_test) if len(X_test) > 0  else None,
            pd.concat(Y_test) if len(Y_test) > 0 else None)

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

def make_suffixes(args):
    alone_suffix = "alone" if args.include_alone else "paironly"
    dropzero_suffix = "dropzero" if args.dropzero else "inczero"
    cutoff_suffix = "{}".format(math.ceil(args.cutoff*100))
    return (alone_suffix, dropzero_suffix, cutoff_suffix)

def make_svm_result_filename(prefix, args, kconfig):
    alone_suffix, dropzero_suffix, cutoff_suffix = make_suffixes(args)

    filename = prefix + "_training_{}_core_{}_uncore_{}_features_{}_{}_{}_{}_{}" \
               .format("_".join(sorted(args.config)), args.core, args.uncore, "_".join(sorted(args.features)),
                       kconfig, alone_suffix, dropzero_suffix, cutoff_suffix)
    return filename

def make_weka_results_filename(prefix, args):
    alone_suffix, dropzero_suffix, cutoff_suffix = make_suffixes(args)
    filename = '{}_training_{}_core_{}_uncore_{}_features_{}_{}_{}_{}.csv'
    return filename.format(prefix, '_'.join(sorted(args.config)), \
                           args.core, args.uncore, "_".join(sorted(args.features)), \
                           alone_suffix, dropzero_suffix, cutoff_suffix)

if __name__ == '__main__':
    parser = get_argument_parser('Get the SVM parameters for a row in the heatmap.')
    parser.add_argument('--weka', dest='weka', action='store_true', default=False, help='Save files for Weka')
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="List or programs to include for the test set.")
    parser.add_argument('--all', dest='all', action='store_true', default=False, help="Use whole data set as training set.")
    args = parser.parse_args()

    if not args.tests:
        runtimes = get_runtime_dataframe(args.data_directory)
        tests = [[x] for x in sorted(runtimes['A'].unique())] # None here means we save the whole matrix as X (no training set)
    elif args.all:
        tests = [[None]]
    else:
        tests = [args.tests] # Pass the tests as a single set


    if args.paper:
        output_directory = os.getcwd()
    else:
        output_directory = os.path.join(args.data_directory, "results_svm")
        os.makedirs(output_directory, exist_ok=True)

    svr = svm.SVC()
    parameters = {
      'kernel': ['poly'],
      'degree': [1, 2],
      'C': [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 20, 30, 40, 50]
    }

    results_table = pd.DataFrame()

    for test in tests:
        X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args, test)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)

        if test != [None]:
            X_test_scaled = min_max_scaler.transform(X_test)
            clf = model_selection.GridSearchCV(svr, parameters, n_jobs=4)
            clf.fit(X_scaled, Y)

            Y_pred = clf.predict(X_test_scaled)

            row = get_svm_metrics(args, test, Y, Y_test, Y_pred)
            row['kernel'] = clf.best_estimator_.kernel
            row['degree'] = clf.best_estimator_.degree
            row['C'] = clf.best_estimator_.C
            row['class_weight'] = 'balanced'
            results_table = results_table.append(row, ignore_index=True)
            print(results_table)

    filename = make_svm_result_filename("svm_scalesmt_results", args, kconfig)
    results_table.to_csv(os.path.join(output_directory, filename + ".csv"), index=False)
