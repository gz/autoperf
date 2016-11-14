#!/usr/bin/env python

import os
import sys
import time
import argparse
import logging

import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify import get_argument_parser
from analyze.classify.svm import matrix_file_name, drop_zero_events
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.util import *

CLASSIFIERS = {
    'ridge': linear_model.Ridge(normalize=True),
    #'lasso': linear_model.Lasso(normalize=True),
    #'lassoA0.1': linear_model.Lasso(alpha=0.1, normalize=True),
    #'lassoA2': linear_model.Lasso(alpha=2, normalize=True),
}

def get_svc_metrics(args, test, Y, Y_test, Y_pred):
    row = {}
    row['Training Configs'] = ' '.join(args.config)
    row['Tested Application'] = ' '.join(test)

    row['Samples Training Total'] = "{}".format(len(Y))
    row['Samples Test Total'] = "{}".format(len(Y_test))

    row['Explained variance score'] = "%.2f" % metrics.explained_variance_score(Y_test, Y_pred)
    row['Mean absolute error'] = "%.2f" % metrics.mean_absolute_error(Y_test, Y_pred)
    row['Mean squared error'] = "%.2f" % metrics.mean_squared_error(Y_test, Y_pred)
    row['Median absolute error'] = "%.2f" % metrics.median_absolute_error(Y_test, Y_pred)
    row['R2 score'] = "%.2f" % metrics.r2_score(Y_test, Y_pred)

    return row

def row_training_and_test_set(args, tests):
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

                    classification = normalized_runtime
                    if B == "Alone" or B == None:
                        if not args.include_alone:
                            logging.debug("Skipping the samples with {} alone".format(A))
                            continue
                        results_path = os.path.join(args.data_directory, config, "{}".format(A))
                    else:
                        results_path = os.path.join(args.data_directory, config, "{}_vs_{}".format(A, B))
                    matrix_file_path = os.path.join(results_path, matrix_file_name(args.uncore, args.features))

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


if __name__ == '__main__':
    parser = get_argument_parser('Get the SVM parameters for a row in the heatmap.')
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="List or programs to include for the test set.")
    args = parser.parse_args()

    if not args.tests:
        runtimes = get_runtime_dataframe(args.data_directory)
        tests = [[x] for x in sorted(runtimes['A'].unique())] # None here means we save the whole matrix as X (no training set)
    else:
        tests = [args.tests] # Pass the tests as a single set

    output_directory = os.path.join(args.data_directory, "results_regression")
    os.makedirs(output_directory, exist_ok=True)
    for kconfig, clf in list(CLASSIFIERS.items()):
        print(("Trying kernel", kconfig))
        results_table = pd.DataFrame()

        for test in tests:
            X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args, test)

            if test != [None]:
                #X_test_scaled = min_max_scaler.transform(X_test)
                clf.fit(X, Y)
                Y_pred = clf.predict(X_test)

                row = get_svc_metrics(args, test, Y, Y_test, Y_pred)
                results_table = results_table.append(row, ignore_index=True)
                print(results_table)

        filename = make_svm_result_filename("regression", args, kconfig)
        results_table.to_csv(os.path.join(output_directory, filename + ".csv"), index=False)
