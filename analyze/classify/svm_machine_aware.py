#!/usr/bin/env python3

import os
import sys
import time
import argparse
import re
import subprocess

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, font_manager

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify.svm import row_training_and_test_set, get_svm_metrics, make_svm_result_filename
from analyze.classify.svm_topk import get_selected_events, make_ranking_filename
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.classify import get_argument_parser
from analyze.util import *

def mkgroup(cfs_ranking_file):
    AUTOPERF_PATH = os.path.join(sys.path[0], "..", "..", "target", "release", "autoperf")
    ret = subprocess.check_output([AUTOPERF_PATH, "mkgroup", "--input", cfs_ranking_file])
    lines = ret.split(os.linesep)
    assert lines[-1] == ''
    return lines[:-1]

if __name__ == '__main__':
    parser = get_argument_parser('Get the SVM parameters when limiting the amount of features.')
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="List or programs to include for the test set.")
    parser.add_argument('--cfs', dest='cfs', type=str, help="Weka file containing reduced, relevant features.")
    args = parser.parse_args()


    if not args.tests:
        runtimes = get_runtime_dataframe(args.data_directory)
        tests = [[x] for x in sorted(runtimes['A'].unique())] # None here means we save the whole matrix as X (no training set)
    else:
        tests = [args.tests] # Pass the tests as a single set

    for kconfig, clf in list(SVM_KERNELS.items()):
        print(("Trying kernel", kconfig))
        results_table = pd.DataFrame()

        for test in tests:
            if not args.cfs:
                cfs_default_file = os.path.join(args.data_directory, make_ranking_filename(test, args))
                if not os.path.exists(cfs_default_file):
                    print(("Can't process {} because we didn't find the CFS file {}".format(A, cfs_default_file)))
                    sys.exit(1)

                event_list = mkgroup(cfs_default_file)
            else:
                event_list = mkgroup(args.cfs)

            X_all, Y, X_test_all, Y_test = row_training_and_test_set(args.data_directory, args.config, test, uncore=args.uncore, features=args.features, cutoff=args.cutoff, include_alone=args.include_alone, drop_zero=args.dropzero)

            X = pd.DataFrame()
            X_test = pd.DataFrame()

            for event in event_list:
                X[event] = X_all[event]
                X_test[event] = X_test_all[event]

            min_max_scaler = preprocessing.MinMaxScaler()
            X_scaled = min_max_scaler.fit_transform(X)
            X_test_scaled = min_max_scaler.transform(X_test)

            clf.fit(X_scaled, Y)
            Y_pred = clf.predict(X_test_scaled)

            row = get_svm_metrics(args, test, Y, Y_test, Y_pred)
            results_table = results_table.append(row, ignore_index=True)

        filename = make_svm_result_filename("svm_results_machine_aware", args, kconfig)
        results_table.to_csv(filename + ".csv", index=False)
