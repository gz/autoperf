#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Display information about a single event.
"""

import sys
import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt, font_manager
plt.style.use([os.path.join(sys.path[0], '..', 'ethplot.mplstyle')])

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..', ".."))
    from analyze import util

def plot_events(df, features, filename, output_dir, title=None):
    fig = plt.figure()
    if title:
        fig.suptitle(title)

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Events observed [count]')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    for feature in features:
        ax1.plot(df[feature], label=feature)

    ax1.xaxis.set_ticks(np.arange(0, len(df), 4))

    val, labels = plt.xticks()
    plt.xticks(val, ["{}".format(x / 4) for x in val])

    ax1.set_ylim(ymin=0.0)
    ax1.legend(loc='best', prop={'size': 8})

    plt.savefig(os.path.join(output_dir, filename  + ".png"), format='png')
    plt.clf()
    plt.close()
    print("Generated file {}".format(filename + ".png"))


def make_plot(from_directory, features):
    df = util.load_as_X(os.path.join(from_directory, 'results.csv'), aggregate_samples = ['mean', 'std', 'max', 'min'], cut_off_nan=True)
    filename = "perf_event_plot_{}".format("_".join(features))
    plot_events(df, features, filename, from_directory)

if __name__ == '__main__':
    pd.set_option('display.max_rows', 37)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 200)

    parser = argparse.ArgumentParser(description="Plot an event counter.")
    parser.add_argument('--resultdir', dest='dir', type=str, help="Result directory of the profile run.", required=True)
    parser.add_argument('--features', dest='features', nargs='+', type=str, help="Which events to plot (add 'AVG.', 'STD.', 'MAX.' or 'MIN.' in front of the event name)", required=True)
    args = parser.parse_args()

    make_plot(args.dir, args.features)
