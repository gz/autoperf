#!/usr/bin/env python3

import os
import sys
import time
import argparse
import re
import subprocess
import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, font_manager

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify.svm import CLASSIFIERS, row_training_and_test_set, get_svm_metrics, make_svm_result_filename
from analyze.classify.svm_topk import get_selected_events, make_ranking_filename
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.classify import get_argument_parser
from analyze.util import *

def mkgroup(args, kconfig, test):
    assert args.ranking != "sffs"

    topk_input_directory = os.path.join(args.data_directory, "results_svm_topk")
    filename = make_svm_result_filename("svm_topk_{}_for_{}".format(args.ranking, "_".join(sorted(test))), args, kconfig)
    svm_topk_result = os.path.join(topk_input_directory, filename + ".csv")

    if not os.path.exists(svm_topk_result):
        logging.error("Can't process {} because we didn't find the SVM topk result file {}".format(' '.join(sorted(test)), svm_topk_result))
        sys.exit(1)

    AUTOPERF_PATH = os.path.join(sys.path[0], "..", "..", "target", "release", "autoperf")
    ret = subprocess.check_output([AUTOPERF_PATH, "mkgroup", "--input", svm_topk_result])
    lines = ret.decode("utf-8").split(os.linesep)

    assert lines[-1] == ''
    return lines[:-1]

if __name__ == '__main__':
    parser = get_argument_parser('Get the SVM parameters when limiting the amount of features.')
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="List or programs to include for the test set.")
    args = parser.parse_args()


    if not args.tests:
        runtimes = get_runtime_dataframe(args.data_directory)
        tests = [[x] for x in sorted(runtimes['A'].unique())] # None here means we save the whole matrix as X (no training set)
    else:
        tests = [args.tests] # Pass the tests as a single set

    for kconfig, clf in list(CLASSIFIERS.items()):
        logging.info("Trying kernel {}".format(kconfig))
        results_table = pd.DataFrame()

        for test in tests:
            event_list = mkgroup(args, kconfig, test)

            X_all, Y, X_test_all, Y_test = row_training_and_test_set(args, test)

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
