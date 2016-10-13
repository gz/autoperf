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
            regex = r"\s+(\d+)\(\s*(\d+)\s+%\)\s+(\d+)\s+(.*)"
            matches = re.match(regex, line)
            if matches:
                fold = int(matches.group(1))
                index = matches.group(3)
                event = matches.group(4)

                row = { 'column_index': index, 'name': event, 'folds': fold }
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

def error_plot(args, test, df):
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
    plt.savefig("{}_{}_rfecv_corr_rank.png".format(test, '_'.join(args.config)), format='png')

if __name__ == '__main__':
    pd.set_option('display.max_rows', 37)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 200)

    parser = argparse.ArgumentParser(description='Get the SVM parameters when limiting the amount of features.')
    parser.add_argument('--cutoff', dest='cutoff', type=float, default=1.15, help="Cut-off for labelling the runs.")
    parser.add_argument('--uncore', dest='uncore', type=str, help="What uncore counters to include.",
                        default='shared', choices=['all', 'shared', 'exclusive', 'none'])
    parser.add_argument('--test', dest='test', type=str, help="Which program to use as a test set.")
    parser.add_argument('--config', dest='config', nargs='+', type=str, help="Which configs to include (L3-SMT, L3-SMT-cores, ...).")
    parser.add_argument('--ranking', dest='ranking', type=str, help="Weka file containing feature rankings.")
    parser.add_argument('--features', dest='features', type=str, help="Weka file containing reduced, relevant features.")
    parser.add_argument('--rfecv-ranking', dest='rfecv', type=str, help="RFECV file containing feature rankings.")

    parser.add_argument('data_directory', type=str, help="Data directory root.")
    args = parser.parse_args()

    # Add features, according to ranking, repeat
    if not args.rfecv:
        relevant_events = get_selected_events(args.features)
        relevant_events = relevant_events[relevant_events.folds >= 1] # Now, really only take relevant ones :P
        ranking_events = get_event_rankings(args.ranking)
        event_list = pd.merge(relevant_events, ranking_events, on='name', sort=True)
        event_list.sort_values(['folds', 'rank'], inplace=True)
    else:
        ranking_events = get_event_rankings(args.ranking)
        relevant_events = pd.read_csv(args.rfecv)
        event_list = pd.merge(relevant_events, ranking_events, on='name', sort=True)
        event_list.sort_values(['rank_y'], inplace=True)

    runtimes = get_runtime_dataframe(args.data_directory)

    if not args.test:
        tests = sorted(runtimes['A'].unique()):
    else:
        tests = args.test

    for test in tests:
        X_all, Y, X_test_all, Y_test = get_training_and_test_set(args, test)

        X = pd.DataFrame()
        X_test = pd.DataFrame()

        results_table = pd.DataFrame()
        for event in event_list.itertuples():
            print event.name
            X[event.name] = X_all[event.name]
            X_test[event.name] = X_test_all[event.name]

            clf = svm.SVC(kernel='linear')
            min_max_scaler = preprocessing.MinMaxScaler()
            X_scaled = min_max_scaler.fit_transform(X)
            X_test_scaled = min_max_scaler.transform(X_test)

            clf.fit(X_scaled, Y)
            Y_pred = clf.predict(X_test_scaled)

            row = get_svm_metrics(args, test, Y, Y_test, Y_pred)
            row['Event'] = event.name
            results_table = results_table.append(row, ignore_index=True)

        results_table = results_table[['Test App', 'Event', 'Samples', 'Error', 'Accuracy', 'Precision/Recall', 'F1 score']]
        print results_table
        error_plot(args, test, results_table)

        #print results_table.to_latex(index=False)
        # TODO: plot
