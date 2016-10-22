import os, sys
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from StringIO import StringIO

from matplotlib import pyplot as plt, font_manager
from matplotlib.colors import Normalize, LinearSegmentedColormap

colors = LinearSegmentedColormap.from_list('seismic', ["#ca0020", "#2ca25f"])

def heatmap(name, data):
    fig, ax = plt.subplots()
    label_font = font_manager.FontProperties(family='Supria Sans', size=10)
    plt.style.use([os.path.join(sys.path[0], '../ethplot.mplstyle')])

    ax.set_xticklabels(data.index)
    ax.set_yticklabels(data.columns)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    ax.tick_params(pad=11)

    plt.setp(ax.get_xticklabels(), fontproperties=label_font)
    plt.setp(ax.get_yticklabels(), fontproperties=label_font)

    c = plt.pcolor(data, cmap = plt.cm.Blues, vmin=0.0, vmax=100)

    colorbar = plt.colorbar(c, drawedges=False)
    colorbar.outline.set_linewidth(0)
    labels = np.arange(0, 110, 25)
    colorbar.set_ticks(labels)
    colorbar.set_ticklabels([  str(i) + "%" for i in range(0, 110, 25) ])

    values = data.as_matrix()
    """
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            #color = 'white' if values[y][x] > 2.3 else 'black'
            color = 'black'
            plt.text(x + 0.5, y + 0.5, '%.2f' % values[y][x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color=color,
                     fontproperties=ticks_font)
    """

    ax.grid(False)
    ax = plt.gca()
    ax.set_frame_on(False)

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    plt.savefig(os.path.join(sys.argv[1], name + ".png"), format='png')
    plt.savefig(os.path.join(sys.argv[1], name + ".pdf"), format='pdf', pad_inches=0.0)

if __name__ == "__main__":
    NAME = "common_events_heatmap"
    raw_data = pd.read_csv(os.path.join(sys.argv[1], "architecture_comparison.csv"), skipinitialspace=True)
    raw_data['name1 total events'] = raw_data['name1 core events'] + raw_data['name1 uncore events']
    raw_data['name2 total events'] = raw_data['name2 core events'] + raw_data['name2 uncore events']
    raw_data['common events'] = raw_data['common core events'] + raw_data['common uncore events']
    raw_data['common events fraction'] = (raw_data['common events'] / raw_data['name1 total events']) * 100
    raw_data = raw_data.sort_values(by=['year1', 'year2'])

    pivot_table = raw_data.pivot_table(index='name1', columns='name2', values='common events fraction')
    values = [
        "Bonnell",
        "NehalemEP",
        "NehalemEX",
        "WestmereEP-SP",
        "WestmereEP-DP",
        "WestmereEX",
        "Jaketown",
        "SandyBridge",
        "IvyBridge",
        "Silvermont",
        "Haswell",
        "IvyBridgeEP",
        "HaswellX",
        "Broadwell",
        "BroadwellDE",
        "Skylake",
        "BroadwellX",
        "Goldmont",
        "KnightsLanding"
    ]
    mi = pd.MultiIndex.from_product([values])
    pivot_table = pivot_table.reindex_axis(mi, 1)
    pivot_table = pivot_table.reindex_axis(mi, 0)

    heatmap(NAME, pivot_table)
