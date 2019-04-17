import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt, font_manager
from matplotlib.colors import LinearSegmentedColormap

colors = LinearSegmentedColormap.from_list('seismic',
                                           ['#ca0020', '#ffffff', '#2a99d6'])

def heatmap(plot_output_dir, data):
    plt.style.use([os.path.join(sys.path[0], "..", 'ethplot.mplstyle')])
    fig, ax = plt.subplots()

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.xlim(0, data.shape[0])
    plt.ylim(0, data.shape[1])

    c = plt.pcolor(data.iloc[::-1], cmap=colors, vmin=-1.0, vmax=1.0)
    colorbar = plt.colorbar(c, ticks=[-1, 0, 1])

    ticks_font = font_manager.FontProperties(family='Decima Mono')
    plt.setp(colorbar.ax.get_yticklabels(), fontproperties=ticks_font)
    plt.savefig(os.path.join(plot_output_dir, 'heatmap.png'), format='png')

def main():
    if len(sys.argv) >= 2:
        plot_output_dir = sys.argv[1]
    else:
        plot_output_dir = '.'

    data_file = os.path.join(plot_output_dir, 'correlation_matrix.csv')

    data = pd.read_csv(data_file, sep='\t', header=0, index_col=0)
    heatmap(plot_output_dir, data)

if __name__ == '__main__':
    main()
