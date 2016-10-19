"""
Joins two weka correlation ranking files for comparison.
"""

import sys
import os
import re
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, font_manager

ticks_font = font_manager.FontProperties(family='Decima Mono')
plt.style.use([os.path.join(sys.path[0], 'ethplot.mplstyle')])

def get_matrix_file(results_path):
    MATRIX_FILE = 'matrix_X_uncore_{}.csv'.format(args.uncore)

    matrix_file = os.path.join(results_path, MATRIX_FILE)
    if os.path.exists(os.path.join(results_path, 'completed')):
        if os.path.exists(matrix_file):
            return matrix_file
        else:
            print "No matrix file ({}) found, run the scripts/pair/matrix_all.py script first!".format(matrix_file)
            sys.exit(1)
    else:
        print "Unfinished directory: {}".format(results_path)
        sys.exit(1)

def plot_events(args, filename, title, df):
    fig = plt.figure()
    if title:
        fig.suptitle(title)

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('Time [250 ms]')
    ax1.set_ylabel('Events observed [count]')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    for feature in args.features:
        ax1.plot(df[feature], label=feature)

    ax1.set_ylim(ymin=0.0)
    ax1.legend()

    plt.setp(ax1.get_xticklabels(), fontproperties=ticks_font)
    plt.setp(ax1.get_yticklabels(), fontproperties=ticks_font)

    plt.savefig(filename  + ".png", format='png')
    plt.clf()
    plt.close()

if __name__ == '__main__':
    pd.set_option('display.max_rows', 37)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 200)

    parser = argparse.ArgumentParser(description="Plot an event counter.")
    parser.add_argument('--run', dest='run', type=str, help="Directory of the run.", required=True)
    parser.add_argument('--features', dest='features', nargs='+', type=str, help="Which features to plot.", required=True)
    parser.add_argument('--uncore', dest='uncore', type=str, help="What uncore counters to include.",
                        default='shared', choices=['all', 'shared', 'exclusive', 'none'])
    args = parser.parse_args()

    path, pair = os.path.split(args.run)
    _, config = os.path.split(path)
    filename = "feature_plot_{}_{}_{}".format(config, pair, "_".join(args.features))

    matrix_file = get_matrix_file(args.run)
    df = pd.read_csv(matrix_file, index_col=False)

    plot_events(args, filename, "{} {}".format(config, pair), df)
