#!/usr/bin/env python3

import os
import sys
import time
import argparse
import re
import subprocess
import logging
from multiprocessing import Pool, TimeoutError, cpu_count

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify.svm import get_svm_metrics, CLASSIFIERS, get_argument_parser, make_svm_result_filename, make_suffixes
from analyze.classify.svm_topk import make_ranking_filename
from analyze.classify.svm_heatmap import cellwise_test_set, rowwise_training_set, generate_heatmaps
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.util import *

def mkgroup(args, kconfig, test):
    assert args.ranking != "sffs"

    topk_input_directory = os.path.join(args.data_directory, "results_svm_topk")
    filename = make_svm_result_filename("svm_topk_{}_for_{}".format(args.ranking, "_".join(sorted(test))), args, kconfig)
    svm_topk_result = os.path.join(topk_input_directory, filename + ".csv")

    if not os.path.exists(svm_topk_result):
        logging.error("Can't process {} because we didn't find the SVM topk result file {}".format(' '.join(sorted(test)), svm_topk_result))
        sys.exit(1)

    df = pd.read_csv(svm_topk_result)
    return df['Event']


def classify(args, kconfig, clf, A, columns, config):
    X = pd.DataFrame()

    event_list = mkgroup(args, kconfig, [A])
    logging.debug("Got events for {}: {}".format(A, ','.join(event_list)))

    cells = []
    X_all, Y = rowwise_training_set(args, A, config)
    for event in event_list:
        X[event] = X_all[event]

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    clf.fit(X_scaled, Y)

    for B in columns:
        if B == "Alone":
            continue

        X_test_all, Y_test, runtime = cellwise_test_set(args, A, B, config)
        X_test = pd.DataFrame()
        for event in event_list:
            X_test[event] = X_test_all[event]

        X_test_scaled = min_max_scaler.transform(X_test)
        Y_pred = clf.predict(X_test_scaled)

        pred = get_svm_metrics(args, [A], Y, Y_test, Y_pred)
        pred['NormalizedRuntime'] = runtime
        pred['A'] = A
        pred['B'] = B
        pred['config'] = config
        cells.append(pred)

    print(pd.DataFrame(cells))
    return pd.DataFrame(cells)

if __name__ == '__main__':
    parser = get_argument_parser("Compute predicition ability for every cell in the heatmap with limited amount of features.",
                                 arguments=['data', 'core', 'uncore', 'cutoff', 'config', 'alone',
                                            'features', 'dropzero', 'ranking', 'kernel', 'paper'])
    args = parser.parse_args()

    if args.paper:
        output_directory = os.getcwd()
    else:
        output_directory = os.path.join(args.data_directory, "results_svm_machine_aware_heatmap")

    os.makedirs(output_directory, exist_ok=True)

    if args.kernel:
        kernels = [ (args.kernel, CLASSIFIERS[args.kernel]) ]
    else:
        kernels = list(CLASSIFIERS.items())

    for kconfig, clf in kernels:
        logging.info("Trying kernel {}".format(kconfig))

        pool = Pool(processes=cpu_count())
        rows = []
        runtimes = get_runtime_dataframe(args.data_directory)
        for config, table in get_runtime_pivot_tables(runtimes):
            if config in args.config:
                for (A, values) in table.iterrows():
                    res = pool.apply_async(classify, (args, kconfig, clf, A, table.columns, config))
                    rows.append(res)

        results_table = pd.concat([r.get() for r in rows], ignore_index=True)
        pool.close()
        pool.join()

        filename = make_svm_result_filename("svm_topk_heatmap_{}".format(args.ranking), args, kconfig)
        results_table.to_csv(filename + ".csv", index=False)
        generate_heatmaps(args, output_directory, filename, results_table, kconfig)
