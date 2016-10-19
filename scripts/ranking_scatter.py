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

def get_selected_events(weka_cfs_ranking_file):
    df = pd.DataFrame()
    with open(weka_cfs_ranking_file) as f:
        for line in f.readlines():
            regex = r"\s+[-+]?([0-9]*\.[0-9]+|[0-9]+)\s+(\d+)\s+(.*)"
            matches = re.match(regex, line)
            if matches:
                rank = float(matches.group(1))
                index = matches.group(2)
                event = matches.group(3)

                row = { 'index': index, 'name': event, 'rank': rank }
                df = df.append(row, ignore_index=True)
    return df

def scatterplot(df, filename):
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('Training rank')
    ax1.set_ylabel('Test rank')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_xlim((0,1))
    ax1.set_ylim((0,1))

    plt.scatter(df['rank_training'], df['rank_test'])
    plt.savefig(filename + ".png", format='png')

    plt.clf()
    plt.close()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 37)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 200)

    parser = argparse.ArgumentParser(description="Merge two ranking files for comparison")
    parser.add_argument('--training', dest='training', type=str, help="Training ranking file.", required=True)
    parser.add_argument('--test', dest='test', type=str, help="Test ranking file.", required=True)
    args = parser.parse_args()

    df_training = get_selected_events(args.training)
    df_test = get_selected_events(args.test)

    df_training.set_index('name')
    df_test.set_index('name')
    del df_training['index']
    del df_test['index']

    filename, file_extension = os.path.splitext(args.training)

    df_merged = pd.merge(df_training, df_test, left_on='name', right_on='name', suffixes=['_training', '_test'])
    df_merged.to_csv(filename + ".csv", index=False)

    df_merged = pd.merge(df_test, df_training, left_on='name', right_on='name', suffixes=['_test', '_training'])
    df_merged.to_csv(args.test + ".csv", index=False)

    scatterplot(df_merged, filename)
