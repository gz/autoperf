#!/usr/bin/env python3

import os
import sys
import time
import argparse
import math

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
    'poly1balancedC0.1': svm.SVC(kernel='poly', degree=1, class_weight='balanced', C=0.1),
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

def drop_zero_events(data_directory, configs, uncore, df):
    from analyze.classify.find_all_zero import zero_features
    to_drop = zero_features(data_directory, configs, uncore, overwrite=False)
    df.drop(to_drop['EVENT_NAME'], axis=1, inplace=True)

def row_training_and_test_set(data_directory, configs, tests, uncore='shared', cutoff=1.15, include_alone=False, drop_zero=True):
    # matrix_X_uncore_shared_aggregation_mean_std_min_max.csv
    MATRIX_FILE = 'matrix_X_uncore_{}_aggregation_mean_std_min_max.csv'.format(uncore)

    X = []
    Y = []
    Y_weights = []

    X_test = []
    Y_test = []

    runtimes = get_runtime_dataframe(data_directory)
    for config, table in get_runtime_pivot_tables(runtimes):
        if config in configs:
            for (A, values) in table.iterrows():
                for (i, normalized_runtime) in enumerate(values):
                    B = table.columns[i]

                    classification = True if normalized_runtime > cutoff else False
                    if B == "Alone":
                        if not include_alone:
                            #print "Skipping the samples with {} alone".format(A)
                            continue
                        results_path = os.path.join(data_directory, config, "{}".format(A))
                    else:
                        results_path = os.path.join(data_directory, config, "{}_vs_{}".format(A, B))
                    matrix_file = os.path.join(results_path, MATRIX_FILE)

                    if os.path.exists(os.path.join(results_path, 'completed')):
                        if not os.path.exists(matrix_file):
                            print("No matrix file ({}) found, run the scripts/pair/matrix_all.py script first!".format(matrix_file))
                            sys.exit(1)
                        df = pd.read_csv(matrix_file, index_col=False)
                        if drop_zero:
                            drop_zero_events(data_directory, configs, uncore, df)

                        if A in tests:
                            #print("Adding {} vs {} to test set".format(A, B), classification)
                            Y_test.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                            X_test.append(df)
                        elif B in tests:
                            #print("Discarding {} vs {}".format(A, B), classification)
                            pass
                        else:
                            #print("Adding {} vs {} to training set".format(A, B), classification)
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

def make_result_filename(prefix, args, kconfig):
    alone_suffix = "alone" if args.include_alone else "paironly"
    cutoff_suffix = "{}".format(math.ceil(args.cutoff*100))
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
        tests = [[x] for x in sorted(runtimes['A'].unique())] # None here means we save the whole matrix as X (no training set)
    else:
        tests = [args.tests] # Pass the tests as a single set

    if not args.weka:
        for kconfig, clf in list(CLASSIFIERS.items()):
            print(("Trying kernel", kconfig))
            results_table = pd.DataFrame()

            for test in tests:
                X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args.data_directory, args.config, test, uncore=args.uncore, cutoff=args.cutoff, include_alone=args.include_alone, drop_zero=True)
                min_max_scaler = preprocessing.MinMaxScaler()
                X_scaled = min_max_scaler.fit_transform(X)

                if test != [None]:
                    X_test_scaled = min_max_scaler.transform(X_test)
                    clf.fit(X_scaled, Y)
                    Y_pred = clf.predict(X_test_scaled)

                    row = get_svm_metrics(args, test, Y, Y_test, Y_pred)
                    results_table = results_table.append(row, ignore_index=True)
                    print(results_table)

            filename = make_result_filename("svm_results", args, kconfig)
            results_table.to_csv(filename + ".csv", index=False)
    elif args.weka:
        for test in tests:
            # TODO: Weka has a bug when the 2nd class appears late in the vector it will think this
            # file has only one class and complain. THe solutionis to make sure both class label appear
            # directly for example as first and 2nd row XD
            X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args.data_directory, args.config, test, uncore=args.uncore, cutoff=args.cutoff, include_alone=args.include_alone, drop_zero=True)

            X['Y'] = Y
            X_test['Y'] = Y_test

            X['Y'] = X['Y'].map(lambda x: 'Y' if x else 'N')
            X_test['Y'] = X_test['Y'].map(lambda x: 'Y' if x else 'N')

            training_file_name = "unset"
            alone_suffix = "alone" if args.include_alone else "paironly"
            cutoff_suffix = "{}".format(math.ceil(args.cutoff*100))
            if test == [None]:
                training_file_name = 'XY_complete_{}_uncore_{}_{}_{}.csv' \
                                     .format('_'.join(args.config), args.uncore, alone_suffix, cutoff_suffix)
            else:
                training_file_name = 'XY_training_without_{}_training_{}_uncore_{}_{}_{}.csv' \
                                     .format('_'.join(test), '_'.join(args.config), args.uncore, alone_suffix, cutoff_suffix)

            X.to_csv(os.path.join(args.data_directory, training_file_name), index=False)
            if test != [None]:
                test_file_name = 'XY_test_{}_training_{}_uncore_{}_{}_{}.csv' \
                                 .format('_'.join(test), '_'.join(args.config), args.uncore, alone_suffix, cutoff_suffix)
                X_test.to_csv(os.path.join(args.data_directory, test_file_name), index=False)
