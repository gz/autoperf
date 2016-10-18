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

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

from svm import row_training_and_test_set, get_svm_metrics

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
    ticks_font = font_manager.FontProperties(family='Decima Mono')
    plt.style.use([os.path.join(sys.path[0], '..', 'ethplot.mplstyle')])
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('Events [Count]')
    ax1.set_ylabel('Error [%]')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    p = ax1.plot(df['Error'], label=test)
    plt.savefig(filename, format='png')


if __name__ == '__main__':
    pd.set_option('display.max_rows', 37)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 200)

    parser = argparse.ArgumentParser(description='Get the SVM parameters when limiting the amount of features.')
    parser.add_argument('--data', dest='data_directory', type=str, help="Data directory root.")

    parser.add_argument('--cutoff', dest='cutoff', type=float, default=1.15, help="Cut-off for labelling the runs.")
    parser.add_argument('--uncore', dest='uncore', type=str, help="What uncore counters to include.",
                        default='shared', choices=['all', 'shared', 'exclusive', 'none'])
    parser.add_argument('--config', dest='config', nargs='+', type=str, help="Which configs to include (L3-SMT, L3-SMT-cores, ...).",
                        default=['L3-SMT'])
    parser.add_argument('--cfs', dest='cfs', type=str, help="Weka file containing reduced, relevant features.")
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="Which programs to use as a test set.")
    parser.add_argument('--alone', dest='include_alone', action='store_true',
                        default=False, help="Include alone runs.")

    args = parser.parse_args()


    if not args.tests:
        runtimes = get_runtime_dataframe(args.data_directory)
        tests = map(lambda x: [x], sorted(runtimes['A'].unique()))
    else:
        tests = [args.tests]

    for test in tests:
        if not args.cfs:
            cfs_default_file = os.path.join(args.data_directory, "weka_{}_cfssubset_greedystepwise_{}.txt"
                .format('_'.join(test), '_'.join(args.config)))
            if not os.path.exists(cfs_default_file):
                print "Skipping {} because we didn't find the cfs file {}".format(' '.join(test), cfs_default_file)
                continue
            event_list = get_selected_events(cfs_default_file)
        else:
            event_list = get_selected_events(args.cfs)

        X_all, Y, X_test_all, Y_test = row_training_and_test_set(args, test)

        X = pd.DataFrame()
        X_test = pd.DataFrame()

        results_table = pd.DataFrame()
        for event in event_list.head(25).itertuples():
            X[event.name] = X_all[event.name]
            X_test[event.name] = X_test_all[event.name]

            clf = svm.SVC(kernel='poly', degree=1, class_weight='balanced')
            min_max_scaler = preprocessing.MinMaxScaler()
            X_scaled = min_max_scaler.fit_transform(X)
            X_test_scaled = min_max_scaler.transform(X_test)

            clf.fit(X_scaled, Y)
            Y_pred = clf.predict(X_test_scaled)

            row = get_svm_metrics(args, test, Y, Y_test, Y_pred)
            row['Event'] = event.name
            results_table = results_table.append(row, ignore_index=True)


        filename = "{}_{}_topk_cfs_greedyranker".format("_".join(test), '_'.join(args.config))
        results_table.to_csv(filename + ".csv", index=False)
        error_plot(args, filename + ".png", results_table)

        #results_table = results_table[['Test App', 'Event', 'Samples', 'Error', 'Accuracy', 'Precision/Recall', 'F1 score']]
        print results_table
