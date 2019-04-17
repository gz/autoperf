#!/usr/bin/env python3
"""
Compute the pairwise correlation for all events in results.csv
and stores it in correlation_matrix.csv.
Also generates a heatmap for the computed matrix 
and stores it in correlation_heatmap.csv.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.colors import LinearSegmentedColormap

colors = LinearSegmentedColormap.from_list('seismic',
                                           ['#ca0020', '#ffffff', '#2a99d6'])

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..', ".."))
    from analyze import util

def correlation_matrix(data_directory):
    df = util.load_as_X(os.path.join(data_directory, 'results.csv'), cut_off_nan=True, remove_zero=True)
    correlation_matrix = df.corr()
    # Ensure all values in correlation matrix are valid
    assert not correlation_matrix.isnull().values.any()

    correlation_file = os.path.join(data_directory, 'correlation_matrix.csv')
    correlation_matrix.to_csv(correlation_file)
    print("Generated correlation_matrix.csv")

def correlation_heatmap(data_directory):
    data_file = os.path.join(data_directory, 'correlation_matrix.csv')
    data = pd.read_csv(data_file, header=0, index_col=0)
    def make_heatmap(plot_output_dir, data):
        plt.style.use([os.path.join(sys.path[0], "..", 'ethplot.mplstyle')])
        fig, ax = plt.subplots()

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.xlim(0, data.shape[0])
        plt.ylim(0, data.shape[1])

        c = plt.pcolor(data.iloc[::-1], cmap=colors, vmin=-1.0, vmax=1.0)
        colorbar = plt.colorbar(c, ticks=[-1, 0, 1])

        #ticks_font = font_manager.FontProperties(family='Decima Mono')
        #plt.setp(colorbar.ax.get_yticklabels(), fontproperties=ticks_font)
        plt.savefig(os.path.join(plot_output_dir, 'correlation_heatmap.png'), format='png')
        print("Generated correlation_heatmap.png")

    make_heatmap(data_directory, data)

def usage(progname):
    print('usage:', progname, '[data_input_dir]')
    sys.exit(0)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        usage(sys.argv[0])
    correlation_matrix(sys.argv[1])
    correlation_heatmap(sys.argv[1])

