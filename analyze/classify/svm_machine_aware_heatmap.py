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

def mkgroup(cfs_ranking_file):
    AUTOPERF_PATH = os.path.join(sys.path[0], "..", "..", "target", "release", "autoperf")
    ret = subprocess.check_output([AUTOPERF_PATH, "mkgroup", "--input", cfs_ranking_file])
    lines = ret.split(os.linesep)
    assert lines[-1] == ''
    return lines[:-1]

def classify(args, clf, A, columns, config):
    X = pd.DataFrame()

    ranking_file = os.path.join(args.data_directory, 'ranking', make_ranking_filename([A], args))
    if not os.path.exists(ranking_file):
        logging.error("Can't process {} because we didn't find the CFS file {}".format(A, ranking_file))
        return None

    event_list = mkgroup(ranking_file)
    logging.debug("Got events", ranking_file)

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

        X_test_all, Y_test = cellwise_test_set(args, A, B, config)
        X_test = pd.DataFrame()
        for event in event_list:
            X_test[event] = X_test_all[event]

        X_test_scaled = min_max_scaler.transform(X_test)
        Y_pred = clf.predict(X_test_scaled)

        pred = get_svm_metrics(args, [A], Y, Y_test, Y_pred)
        pred['A'] = A
        pred['B'] = B
        pred['config'] = config
        cells.append(pred)

    logging.info(pd.DataFrame([row]))
    return pd.DataFrame([row])

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
                    res = pool.apply_async(classify, (args, clf, A, table.columns, config))
                    rows.append(res)

        results_table = pd.concat([r.get() for r in rows], ignore_index=True)
        pool.close()
        pool.join()

        filename = make_svm_result_filename("svm_machine_aware_heatmap", args, kconfig)
        results_table.to_csv(filename + ".csv", index=False)
        generate_heatmaps(args, output_directory, filename, results_table, kconfig)
