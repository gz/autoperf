#!/usr/bin/env python

import os
import sys
import time
import argparse
import re

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, font_manager

from runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from util import *
from svm import get_argument_parser

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

from svm import SVM_KERNELS, row_training_and_test_set, get_svm_metrics, make_result_filename

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

                row = { 'index': index, 'name': event, 'goodness': goodness }
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
    This is similar to other classify methods but it will reduced
    the X and X_test to only the events listed in event_list.
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

if __name__ == '__main__':
    parser = get_argument_parser('Get the SVM parameters when limiting the amount of features.')
    parser.add_argument('--cfs', dest='cfs', type=str, help="Weka file containing reduced, relevant features.")
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="List or programs to include for the test set.")
    args = parser.parse_args()

    if not args.tests:
        runtimes = get_runtime_dataframe(args.data_directory)
        tests = map(lambda x: [x], sorted(runtimes['A'].unique()))
    else:
        tests = [args.tests]

    for kconfig, clf in SVM_KERNELS.iteritems():
        print "Trying kernel", kconfig
        results_table = pd.DataFrame()

        for test in tests:
            if not args.cfs:
                cfs_default_file = os.path.join(args.data_directory, "wekanew", "weka_{}_cfssubset_greedystepwise_{}.txt"
                    .format('_'.join(test), '_'.join(args.config)))
                if not os.path.exists(cfs_default_file):
                    print "Skipping {} because we didn't find the cfs file {}".format(' '.join(test), cfs_default_file)
                    continue
                event_list = get_selected_events(cfs_default_file)
            else:
                event_list = get_selected_events(args.cfs)

            results_table = classify(args, test, clf, event_list)

            filename = make_result_filename("svm_topk_for_{}".format("_".join(test)), args, kconfig)
            results_table.to_csv(filename + ".csv", index=False)
            error_plot(args, filename, results_table)
