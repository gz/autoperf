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

from svm import get_training_and_test_set, get_svm_metrics

def get_selected_events(weka_fold_file):
    df = pd.DataFrame()
    with open(weka_fold_file) as f:
        for line in f.readlines():
            regex = r"\s+ [-+]?([0-9]*\.[0-9]+|[0-9]+)\s+(\d+)\s+(.*)"
            matches = re.match(regex, line)
            if matches:
                goodness = float(matches.group(1))
                index = matches.group(2)
                event = matches.group(3)

                row = { 'index': index, 'name': event, 'goodness': goodness }
                df = df.append(row, ignore_index=True)
    return df

def get_event_rankings(weka_rankings_file):
    df = pd.DataFrame()
    with open(weka_rankings_file) as f:
        index = 0
        for line in f.readlines():
            splits = line.strip().split(' ')
            # Assumes the event name is last in the line and starts with either AVG. or STD.
            if splits[-1].startswith("AVG.") or splits[-1].startswith("STD."):
                event = splits[-1]
                row = { 'name': event, 'rank': index }
                df = df.append(row, ignore_index=True)
                index += 1
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
    parser.add_argument('--cutoff', dest='cutoff', type=float, default=1.15, help="Cut-off for labelling the runs.")
    parser.add_argument('--uncore', dest='uncore', type=str, help="What uncore counters to include.",
                        default='shared', choices=['all', 'shared', 'exclusive', 'none'])
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="Which programs to use as a test set.")
    parser.add_argument('--config', dest='config', nargs='+', type=str, help="Which configs to include (L3-SMT, L3-SMT-cores, ...).")
    parser.add_argument('--cfs', dest='cfs', type=str, help="Weka file containing reduced, relevant, ranked features using the CFS method.")

    parser.add_argument('data_directory', type=str, help="Data directory root.")
    args = parser.parse_args()

    # Add features, according to ranking, repeat
    relevant_events = get_selected_events(args.cfs)
    #event_list = relevant_events[relevant_events.folds >= 5] # Now, really only take relevant ones :P
    #event_list.sort_values(['folds'], inplace=True)

    runtimes = get_runtime_dataframe(args.data_directory)

    if not args.tests:
        tests = map(lambda x: [x], sorted(runtimes['A'].unique()))
    else:
        tests = [args.tests]

    for test in tests:
        print "Testing", test
        X_all, Y, X_test_all, Y_test = get_training_and_test_set(args, test)
        results_table = pd.DataFrame()

        X = pd.DataFrame()
        X_test = pd.DataFrame()

        available_event_set = event_list.copy()

        while len(available_event_set) > 0:
            min_error = (100.0, None)

            # Find the event that gives the least error:
            for event in available_event_set.itertuples():
                #print event.name
                X[event.name] = X_all[event.name]
                X_test[event.name] = X_test_all[event.name]

                clf = svm.SVC(kernel='linear')
                min_max_scaler = preprocessing.MinMaxScaler()
                X_scaled = min_max_scaler.fit_transform(X)
                X_test_scaled = min_max_scaler.transform(X_test)

                clf.fit(X_scaled, Y)
                Y_pred = clf.predict(X_test_scaled)
                error = (1.0 - metrics.accuracy_score(Y_test, Y_pred))
                if error < min_error[0]:
                    print "Found new min error", error, event.name
                    min_error = (error, event.name)
                # Remove again and test the next event
                del X[event.name]
                del X_test[event.name]

            print "Selected", min_error
            # Add the event with minimum error to the set:
            X[min_error[1]] = X_all[min_error[1]]
            X_test[min_error[1]] = X_test_all[min_error[1]]

            # Update statistics
            clf = svm.SVC(kernel='linear')
            min_max_scaler = preprocessing.MinMaxScaler()
            X_scaled = min_max_scaler.fit_transform(X)
            X_test_scaled = min_max_scaler.transform(X_test)

            clf.fit(X_scaled, Y)
            Y_pred = clf.predict(X_test_scaled)

            row = get_svm_metrics(args, test, Y, Y_test, Y_pred)
            row['Event'] = min_error[1]
            results_table = results_table.append(row, ignore_index=True)

            print available_event_set
            available_event_set = available_event_set[available_event_set.name != min_error[1]]


        results_table = results_table[['Test App', 'Event', 'Samples', 'Error', 'Accuracy', 'Precision/Recall', 'F1 score']]
        print results_table

        filename = "{}_{}_topk_min_folds".format("_".join(test), '_'.join(args.config))

        error_plot(args, filename + ".png", results_table)
        results_table.to_csv(filename + ".csv", index=False)
