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

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify import get_argument_parser
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.classify.generate_matrix import matrix_file_name
from analyze.util import *

CLASSIFIERS = {
    #'linear': svm.SVC(kernel='linear'),
    #'linearbalanced': svm.SVC(kernel='linear', class_weight='balanced'),
    #'rbf1': svm.SVC(kernel='rbf', degree=1),
    #'rbf1balanced': svm.SVC(kernel='rbf', degree=1, class_weight='balanced'),
    #'poly1': svm.SVC(kernel='poly', degree=1),
    #'poly2': svm.SVC(kernel='poly', degree=2),
    #'poly3': svm.SVC(kernel='poly', degree=3),
    #'poly1balanced': svm.SVC(kernel='poly', degree=1, class_weight='balanced'),
    #'poly2balanced': svm.SVC(kernel='poly', degree=2, class_weight='balanced'),
    #'poly1balancedC0.01': svm.SVC(kernel='poly', degree=1, class_weight='balanced', C=0.01),
    #'poly1balancedC0.05': svm.SVC(kernel='poly', degree=1, class_weight='balanced', C=0.05),
    #'poly1balancedC0.075': svm.SVC(kernel='poly', degree=1, class_weight='balanced', C=0.075),
    'poly1balancedC0.1': svm.SVC(kernel='poly', degree=1, class_weight='balanced', C=0.1),
    #'poly1balancedC0.2': svm.SVC(kernel='poly', degree=1, class_weight='balanced', C=0.2),
    #'poly1balancedC0.3': svm.SVC(kernel='poly', degree=1, class_weight='balanced', C=0.3),
    #'poly1balancedC10': svm.SVC(kernel='poly', degree=1, class_weight='balanced', C=10),
    #'poly1balancedC100': svm.SVC(kernel='poly', degree=1, class_weight='balanced', C=100),
    #'poly2balancedC0.1': svm.SVC(kernel='poly', degree=2, class_weight='balanced', C=0.1),
    #'poly2balancedC10': svm.SVC(kernel='poly', degree=2, class_weight='balanced', C=10),
    #'poly2balancedC100': svm.SVC(kernel='poly', degree=2, class_weight='balanced', C=100),
    #'poly3balanced': svm.SVC(kernel='poly', degree=3, class_weight='balanced'),
    #'neural': neural_network.MLPClassifier(),
    #'neuralsgd': neural_network.MLPClassifier(solver='sgd'),
    #'neuraladaptivelogistic': neural_network.MLPClassifier(activation='logistic', learning_rate='adaptive'),
    #'passiveaggr': linear_model.PassiveAggressiveClassifier(),
    #'randomforest10': ensemble.RandomForestClassifier(n_estimators=10),
    #'decision': tree.DecisionTreeClassifier(),
    #'decision5': tree.DecisionTreeClassifier(max_features=5),
    #'decision10': tree.DecisionTreeClassifier(max_features=10),
    #'decision15': tree.DecisionTreeClassifier(max_features=15),
    #'decision20': tree.DecisionTreeClassifier(max_features=20),
    #'decision25': tree.DecisionTreeClassifier(max_features=25),
    #'kneighbors': neighbors.KNeighborsClassifier(),
    #'kneighborsdistance': neighbors.KNeighborsClassifier(weights='distance'),
    #'adaboost': ensemble.AdaBoostClassifier()
}

def drop_zero_events(args, df):
    from analyze.classify.find_all_zero import zero_features
    to_drop = zero_features(args, overwrite=False)
    df.drop(to_drop['EVENT_NAME'], axis=1, inplace=True)

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

                    classification = True if normalized_runtime > args.cutoff else False
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
                            logging.debug("Dropping zero")
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
    filename = prefix + "_training_{}_uncore_{}_features_{}_{}_{}_{}_{}" \
               .format("_".join(sorted(args.config)), args.uncore, "_".join(sorted(args.features)),
                       kconfig, alone_suffix, dropzero_suffix, cutoff_suffix)
    return filename

def make_weka_results_filename(prefix, args):
    alone_suffix, dropzero_suffix, cutoff_suffix = make_suffixes(args)
    filename = '{}_training_{}_uncore_{}_features_{}_{}_{}.csv'
    return filename.format(prefix, '_'.join(sorted(args.config)), \
                           args.uncore, "_".join(sorted(args.features)), \
                           alone_suffix, dropzero_suffix, cutoff_suffix)

if __name__ == '__main__':
    parser = get_argument_parser('Get the SVM parameters for a row in the heatmap.')
    parser.add_argument('--weka', dest='weka', action='store_true', default=False, help='Save files for Weka')
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="List or programs to include for the test set.")
    args = parser.parse_args()

    if not args.tests:
        runtimes = get_runtime_dataframe(args.data_directory)
        tests = [[x] for x in sorted(runtimes['A'].unique())] # None here means we save the whole matrix as X (no training set)
    else:
        tests = [args.tests] # Pass the tests as a single set

    if not args.weka:
        for kconfig, clf in list(CLASSIFIERS.items()):
            print(("Trying kernel", kconfig))
            results_table = pd.DataFrame()

            for test in tests:
                X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args, test)
                min_max_scaler = preprocessing.MinMaxScaler()
                X_scaled = min_max_scaler.fit_transform(X)

                if test != [None]:
                    X_test_scaled = min_max_scaler.transform(X_test)
                    clf.fit(X_scaled, Y)
                    Y_pred = clf.predict(X_test_scaled)

                    row = get_svm_metrics(args, test, Y, Y_test, Y_pred)
                    results_table = results_table.append(row, ignore_index=True)
                    print(results_table)

            filename = make_svm_result_filename("svm_results", args, kconfig)
            results_table.to_csv(filename + ".csv", index=False)
    elif args.weka:
        os.makedirs(os.path.join(args.data_directory, "matrices"), exist_ok=True)

        for test in tests:
            # TODO: Weka has a bug when the 2nd class appears late in the vector it will think this
            # file has only one class and complain. THe solutionis to make sure both class label appear
            # directly for example as first and 2nd row XD
            X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args, test)

            X['Y'] = Y
            X_test['Y'] = Y_test

            X['Y'] = X['Y'].map(lambda x: 'Y' if x else 'N')
            X_test['Y'] = X_test['Y'].map(lambda x: 'Y' if x else 'N')

            if test == [None]:
                training_file_name = make_weka_results_filename('XY_complete', args)
            else:
                training_file_name = make_weka_results_filename('XY_training_without_{}'.format('_'.join(sorted(test))), args)
            X.to_csv(os.path.join(args.data_directory, "matrices", training_file_name), index=False)

            if test != [None]:
                test_file_name = make_weka_results_filename('XY_test_{}'.format('_'.join(sorted(test))), args)
                X_test.to_csv(os.path.join(args.data_directory, "matrices", test_file_name), index=False)
