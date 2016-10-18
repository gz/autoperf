#!/usr/bin/env python

import os
import sys
import time
import argparse
import re
import subprocess
from multiprocessing import Pool, TimeoutError

import pandas as pd
import numpy as np

from runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from util import *

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

from svm import get_svm_metrics, SVM_KERNELS, get_argument_parser
from svm_topk import get_selected_events
from svm_heatmap import get_training_and_test_set, get_pivot_tables, heatmap

AUTOPERF_PATH = os.path.join(sys.path[0], "..", "..", "target", "release", "autoperf")

def mkgroup(cfs_ranking_file):
    ret = subprocess.check_output([AUTOPERF_PATH, "mkgroup", "--input", cfs_ranking_file])
    lines = ret.split(os.linesep)
    assert lines[-1] == ''
    return lines[:-1]

def classify(args, clf, A, B, config):
    X_all, Y, X_test_all, Y_test = get_training_and_test_set(args, A, B, config)

    X = pd.DataFrame()
    X_test = pd.DataFrame()

    cfs_default_file = os.path.join(args.data_directory, "{}_{}_topk_cfs_greedyranker.csv".format(A, '_'.join(args.config)))
    if not os.path.exists(cfs_default_file):
        print "Can't process {} because we didn't find the CFS file {}".format(A, cfs_default_file)
        return None

    event_list = mkgroup(cfs_default_file)

    for event in event_list:
        X[event] = X_all[event]
        X_test[event] = X_test_all[event]

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    X_test_scaled = min_max_scaler.transform(X_test)

    clf.fit(X_scaled, Y)
    Y_pred = clf.predict(X_test_scaled)

    row = get_svm_metrics(args, [A], Y, Y_test, Y_pred)
    row['A'] = A
    row['B'] = B
    row['config'] = config

    print pd.DataFrame([row])
    return pd.DataFrame([row])

if __name__ == '__main__':

    parser = get_argument_parser("Compute predicition ability for every cell in the heatmap with limited amount of features.")
    args = parser.parse_args()

    for kconfig, clf in SVM_KERNELS.iteritems():
        print "Trying kernel", kconfig

        pool = Pool(processes=6)
        rows = []
        runtimes = get_runtime_dataframe(args.data_directory)
        for config, table in get_runtime_pivot_tables(runtimes):
            if config in args.config:
                for (A, values) in table.iterrows():
                    for (i, normalized_runtime) in enumerate(values):
                        B = table.columns[i]
                        if B == "Alone":
                            continue
                        res = pool.apply_async(classify, (args, clf, A, B, config))
                        rows.append(res)

        results_table = pd.concat(map(lambda r: r.get(), rows), ignore_index=True)
        pool.close()
        pool.join()

        alone_suffix = "alone" if args.alone else "paironly"
        cutoff_suffix = "{}".format(args.cutoff*100)

        filename = "svm_machine_aware_heatmap_training_{}_uncore_{}_{}_{}_{}" \
                   .format("_".join(args.config), args.uncore, kconfig, alone_suffix, cutoff_suffix)
        results_table.to_csv(filename + ".csv", index=False)

        for (config, pivot_table) in get_pivot_tables(results_table):
            plot_filename = filename + "_config_{}".format(config)
            title = "Machine Aware, Training {}, uncore {}, config {}, kernel {}, {}, {}" \
                    .format("/".join(args.config), args.uncore, config, kconfig, alone_suffix, cutoff_suffix)
            heatmap(filename, pivot_table, title)
