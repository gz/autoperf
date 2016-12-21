#!/usr/bin/env python3

import os
import sys
import time
import argparse
import re
import logging
from multiprocessing import Pool, TimeoutError, cpu_count

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

import matplotlib
matplotlib.rc('pdf', fonttype=42)
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

def error_plot_all(args, output_directory, results, baseline_results):
    fig, axarr = plt.subplots(5, 2, sharex='col', sharey='row', figsize=(13, 10))
    plt.subplots_adjust(left=None, bottom=0.0, right=None, top=10.0, wspace=5.0, hspace=5.0)

    index = 0
    for test, filename, df in results:
        row = math.floor(index / 2)
        col = index % 2
        ax = axarr[row][col]
        index += 1

        if col == 0:
            ax.set_ylabel('Error [%]')
        if row == 4:
            ax.set_xlabel('Features [Count]')

        assert(len(test) == 1)
        ax.set_title(test[0].strip(), loc='right', fontsize=18, position=(0.99, 0.99))
        ax.set_ylim((0.0, 1.0))
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks([1, 5, 10, 15, 20, 25])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        #ax.annotate(' '.join(test), xy=(24.5, 0.95), size=14, ha='right', va='top')
        df.index += 1

        if baseline_results is not None:
            assert(len(test) == 1)
            row = baseline_results[baseline_results['Tested Application'] == test[0]]
            bl = ax.axhline(y=row.Error.values[0], xmin=0, xmax=1, color="#fc4f30")

        p = ax.plot(df['Error'], linewidth=2, label="Reduced Features")
        [line.set_zorder(3) for line in ax.lines]
        #ax.legend(fontsize=15, loc=(0.54, 0.7)) # loc='upper right'

    fig.legend((p[0], bl), ("Reduced Features", "Baseline (All Features)"),
               loc=(0.47, 0.96), ncol=2, fontsize=20)

    filename = make_svm_result_filename("svm_topk_{}_for_all".format(args.ranking), args, kconfig)
    location = os.path.join(output_directory, filename)
    plt.tight_layout()
    plt.savefig(location + ".pdf", format='pdf', pad_inches=0.0)
    plt.clf()
    plt.close()

def error_plot(args, test, output_directory, filename, df, baseline_results=None):
    fig = plt.figure()
    if not args.paper:
        fig.suptitle(filename)

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('Events [Count]')
    ax1.set_ylabel('Error [%]')
    ax1.set_ylim((0.0, 1.0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    p = ax1.plot(df['Error'], label=test[0])

    # Add the base line to the plot:
    if baseline_results is not None:
        assert(len(test) == 1)
        row = baseline_results[baseline_results['Tested Application'] == test[0]]
        bl = ax1.axhline(y=row.Error.values[0], xmin=0, xmax=1, color="#fc4f30", label="Baseline (All Features)")

    # Add the first 5 event names to the plot:
    y_text = np.arange(min(df['Error']), max(df['Error']), 0.06)
    for idx, name in enumerate(df['Event']):
        if idx < 5:
            displacement = 0.02
            displacement_x = 1.50
            scale_x = 1
            ax1.annotate(name, xy=(idx+1, df['Error'].iloc[idx]), xytext=((idx/scale_x)+displacement_x, y_text[-idx-1]+displacement),
                         color=p[0].get_color(),
                         arrowprops=dict(edgecolor="#999999", facecolor="#999999",
                                         width=0.7, headwidth=5, headlength=8))

    location = os.path.join(output_directory, filename)
    if args.paper:
        plt.savefig(location + ".pdf", format='pdf', pad_inches=0.0)

    plt.savefig(location  + ".png", format='png')
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
        if args.ranking == 'sffs':
            from ast import literal_eval as make_tuple
            X = pd.DataFrame()
            X_test = pd.DataFrame()
            for (idx, feature) in enumerate(make_tuple(event.feature_idx)):
                X[idx] = X_all[X_all.columns[feature]]
                X_test[idx] = X_test_all[X_test_all.columns[feature]]
                print(idx, X_all.columns[feature])
        else:
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

def evaluate_test(args, output_directory, kconfig, test, clf, event_list, baseline_results):
    results_table = classify(args, test, clf, event_list)
    filename = make_svm_result_filename("svm_topk_{}_for_{}".format(args.ranking, "_".join(sorted(test))), args, kconfig)

    results_table.to_csv(os.path.join(output_directory, filename + ".csv"), index=False)
    error_plot(args, test, output_directory, filename, results_table, baseline_results)

    return (test, filename, results_table)


if __name__ == '__main__':
    parser = get_argument_parser('Get the SVM parameters when limiting the amount of features.',
                                 arguments=['data', 'core', 'uncore', 'cutoff', 'config', 'alone',
                                            'features', 'dropzero', 'ranking', 'kernel', 'paper'])
    parser.add_argument('--tests', dest='tests', nargs='+', type=str, help="List or programs to include for the test set.")
    args = parser.parse_args()

    if args.paper:
        output_directory = os.getcwd()
    else:
        output_directory = os.path.join(args.data_directory, "results_svm_topk")
    os.makedirs(output_directory, exist_ok=True)

    if args.kernel:
        kernels = [ (args.kernel, CLASSIFIERS[args.kernel]) ]
    else:
        kernels = list(CLASSIFIERS.items())

    if not args.tests:
        runtimes = get_runtime_dataframe(args.data_directory)
        tests = [[x] for x in sorted(runtimes['A'].unique())]
    else:
        tests = [args.tests]

    for kconfig, clf in kernels:
        logging.info("Trying kernel {}".format(kconfig))

        baseline_results_filename = make_svm_result_filename("svm_results", args, kconfig) + ".csv"
        if os.path.exists(baseline_results_filename):
            baseline_results = pd.read_csv(baseline_results_filename)
        else:
            baseline_results = None

        pool = Pool(processes=cpu_count())
        rows = []
        for test in tests:
            cfs_default_file = os.path.join(args.data_directory, 'ranking', make_ranking_filename(test, args))
            if not os.path.exists(cfs_default_file):
                logging.warn("Skipping {} because we didn't find the ranking file: {}".format(' '.join(test), cfs_default_file))
                continue
            if args.ranking != 'sfs' and args.ranking != 'sffs':
                event_list = get_selected_events(cfs_default_file)
            else:
                event_list = pd.read_csv(cfs_default_file)

            res = pool.apply_async(evaluate_test, (args, output_directory, kconfig, test, clf, event_list, baseline_results))
            rows.append(res)

        results = [r.get() for r in rows]

        #results = [ (test, "xxx", pd.DataFrame(np.random.randint(0, 2, size=(25, 1)), columns=['Error'])) for test in tests ]
        error_plot_all(args, output_directory, results, baseline_results)

        pool.close()
        pool.join()
