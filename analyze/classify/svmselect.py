#!/usr/bin/env python3

import os
import sys
import time
import argparse
import math
import logging
from multiprocessing import cpu_count

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
from analyze.classify.svm import row_training_and_test_set, get_svm_metrics, make_svm_result_filename
from analyze.util import *

if __name__ == '__main__':
    parser = get_argument_parser('Get the SVM parameters for a row in the heatmap.')
    args = parser.parse_args()

    runtimes = get_runtime_dataframe(args.data_directory)
    tests = [[x] for x in sorted(runtimes['A'].unique())] # None here means we save the whole matrix as X (no training set)

    output_directory = os.getcwd()
    results_table = pd.DataFrame()

    svr = svm.SVC(class_weight='balanced')
    parameters = {
      'kernel': ['poly'],
      'degree': [1, 2],
      'C': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }

    for test in tests:
        X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args, test)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)

        clf = model_selection.GridSearchCV(svr, parameters, n_jobs=cpu_count())

        if test != [None]:
            X_test_scaled = min_max_scaler.transform(X_test)

            clf.fit(X_scaled, Y)
            #print (test, clf.best_estimator_)

            Y_pred = clf.predict(X_test_scaled)

            row = get_svm_metrics(args, test, Y, Y_test, Y_pred)
            row['kernel'] = clf.best_estimator_.kernel
            row['degree'] = clf.best_estimator_.degree
            row['C'] = clf.best_estimator_.C
            row['class_weight'] = 'balanced'

            results_table = results_table.append(row, ignore_index=True)
            print(results_table)

    filename = make_svm_result_filename("svmselect_results", args, "")
    results_table.to_csv(os.path.join(output_directory, filename + ".csv"), index=False)
