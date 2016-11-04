#!/usr/bin/env python3

import os
import sys
import time
import argparse
import re
import subprocess
from multiprocessing import Pool, TimeoutError

import pandas as pd
import numpy as np



from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify.svm import get_svm_metrics, CLASSIFIERS, get_argument_parser, make_svm_result_filename, make_suffixes
from analyze.classify.svm_topk import make_ranking_filename
from analyze.classify.svm_heatmap import cellwise_test_set, rowwise_training_set, get_pivot_tables, heatmap
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.util import *

def mkgroup(cfs_ranking_file):
    AUTOPERF_PATH = os.path.join(sys.path[0], "..", "..", "target", "release", "autoperf")
    ret = subprocess.check_output([AUTOPERF_PATH, "mkgroup", "--input", cfs_ranking_file])
    lines = ret.split(os.linesep)
    assert lines[-1] == ''
    return lines[:-1]

def classify(args, clf, A, B, config):
    assert "TODO fix this"
    X_all, Y, X_test_all, Y_test = cellwise_training_and_test_set(args, A, B, config)

    X = pd.DataFrame()
    X_test = pd.DataFrame()

    cfs_default_file = os.path.join(args.data_directory, make_ranking_filename([A], args))
    if not os.path.exists(cfs_default_file):
        print(("Can't process {} because we didn't find the CFS file {}".format(A, cfs_default_file)))
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

    print((pd.DataFrame([row])))
    return pd.DataFrame([row])

if __name__ == '__main__':
    parser = get_argument_parser("Compute predicition ability for every cell in the heatmap with limited amount of features.")
    args = parser.parse_args()

    for kconfig, clf in list(CLASSIFIERS.items()):
        print(("Trying kernel", kconfig))

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

        results_table = pd.concat([r.get() for r in rows], ignore_index=True)
        pool.close()
        pool.join()

        filename = make_svm_result_filename("svm_machine_aware_heatmap", args, kconfig)
        results_table.to_csv(filename + ".csv", index=False)

        for (config, pivot_table) in get_pivot_tables(results_table):
            plot_filename = filename + "_config_{}".format(config)

            alone_suffix, dropzero_suffix, cutoff_suffix = make_suffixes(args)
            title = "MAware, Training {}, uncore {}, features {}, config {}, kernel {}, {}, {}, {}" \
                    .format("/".join(sorted(args.config)), args.uncore, " ".join(sorted(args.features)), \
                            config, kconfig, alone_suffix, cutoff_suffix, dropzero_suffix)

            heatmap(plot_filename, pivot_table, title)
