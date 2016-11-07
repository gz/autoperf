#!/usr/bin/env python3

import os
import sys
import time
import argparse
import re
import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, font_manager

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify.svm import CLASSIFIERS, row_training_and_test_set, get_svm_metrics, make_weka_results_filename, make_svm_result_filename
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.classify.svm import get_argument_parser
from analyze.util import *

ticks_font = font_manager.FontProperties(family='Decima Mono')
plt.style.use([os.path.join(sys.path[0], '..', 'ethplot.mplstyle')])

def get_selected_events(weka_cfs_ranking_file):
    df = pd.DataFrame()
    with open(weka_cfs_ranking_file) as f:
        for line in f.readlines():
            regex = r"\s+[-+]?([0-9]*\.[0-9]+|[0-9]+)\s+(\d+)\s+(.*)"
            matches = re.match(regex, line)
            if matches:
                goodness = float(matches.group(1))
                index = matches.group(2)
                event = matches.group(3)

                row = { 'index': int(index)-1, 'name': event, 'goodness': goodness }
                df = df.append(row, ignore_index=True)
    return df

def error_plot(args, filename, df):
    fig = plt.figure()
    fig.suptitle(filename)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('Events [Count]')
    ax1.set_ylabel('Error [%]')
    ax1.set_ylim((0.0, 1.0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    p = ax1.plot(df['Error'], label=test)
    plt.savefig(filename  + ".png", format='png')
    plt.clf()
    plt.close()

def classify(args, test, clf, event_list):
    """
    This is similar to the svm.py classify method but it adds one feature after another
    (specified in event_list) to the classifier and measures its performance.
    """
    X_all, Y, Y_weights, X_test_all, Y_test = row_training_and_test_set(args, test)

    X = pd.DataFrame()
    X_test = pd.DataFrame()

    results_table = pd.DataFrame()
    for event in event_list.head(25).itertuples():
        X[event.name] = X_all[event.name]
        X_test[event.name] = X_test_all[event.name]

        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        X_test_scaled = min_max_scaler.transform(X_test)

        clf.fit(X_scaled, Y)
        Y_pred = clf.predict(X_test_scaled)

        row = get_svm_metrics(args, test, Y, Y_test, Y_pred)
        row['Event'] = event.name
        results_table = results_table.append(row, ignore_index=True)

    return results_table

def make_ranking_filename(apps, args):
    prefix = 'ranking_{}_{}'.format(args.ranking, "_".join(sorted(apps)))
    return make_weka_results_filename(prefix, args)

if __name__ == '__main__':
    parser = get_argument_parser('Get the SVM parameters when limiting the amount of features.')
    parser.add_argument('--cfs', dest='cfs', type=str, help="Weka file containing reduced, relevant features.")
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="List or programs to include for the test set.")
    args = parser.parse_args()

    if not args.tests:
        runtimes = get_runtime_dataframe(args.data_directory)
        tests = [[x] for x in sorted(runtimes['A'].unique())]
    else:
        tests = [args.tests]

    for kconfig, clf in list(CLASSIFIERS.items()):
        print(("Trying kernel", kconfig))

        for test in tests:
            if not args.cfs:
                cfs_default_file = os.path.join(args.data_directory, 'ranking', make_ranking_filename(test, args))
                if not os.path.exists(cfs_default_file):
                    logging.warn("Skipping {} because we didn't find the ranking file: {}".format(' '.join(test), cfs_default_file))
                    continue
                event_list = get_selected_events(cfs_default_file)
            else:
                event_list = get_selected_events(args.cfs)

            results_table = classify(args, test, clf, event_list)

            filename = make_svm_result_filename("svm_topk_for_{}".format("_".join(sorted(test))), args, kconfig)
            results_table.to_csv(filename + ".csv", index=False)
            error_plot(args, filename, results_table)
