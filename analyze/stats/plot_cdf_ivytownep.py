import os, sys
import glob

from matplotlib import pyplot as plt, font_manager
import numpy as np
import pandas as pd

import matplotlib

ticks_font = font_manager.FontProperties(family='Decima Mono')
matplotlib.rc('pdf', fonttype=42)
plt.style.use([os.path.join(sys.path[0], '../ethplot.mplstyle')])

def find_closest(items, val):
    last = 0.0
    for x, y in items:
        print(x, y)
        if y > val:
            return last
        last = y

    return last

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_ylabel('CDF [%]')
    ax1.set_xlabel('Event description [Levenshtein distance]')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.setp(ax1.get_xticklabels(), fontproperties=ticks_font)
    plt.setp(ax1.get_yticklabels(), fontproperties=ticks_font)

    for idx, f in enumerate(glob.glob(os.path.join(sys.argv[1], 'editdist_IvyBridgeEP-vs-*.csv'))):
        filename, file_extension = os.path.splitext(os.path.basename(f))
        line_label = filename.split("vs-")[1].split("\.")[0]

        raw_data = pd.read_csv(f, skipinitialspace=True)
        series = raw_data['edit distance']
        series = series.sort_values()
        series[len(series)] = series.iloc[-1]
        cum_dist = np.linspace(0.,1.,len(series))
        cdf = pd.Series(cum_dist, index=series)
        items = list(cdf.items())
        x, y = idx*10, find_closest(items, idx*0.1)

        if line_label == "KnightsLanding": #not in ["KnightsLanding", "NehalemEP", "IvyBridgeEP", "Skylake", "Goldmont", "Broadwell" ]:
            x = 83
            y = find_closest(items, 0.8)
        elif line_label == "NehalemEP":
            x = 150
            y = find_closest(items, 0.9)
        elif line_label == "Goldmont":
            x = 30
            y = 0.18
        elif line_label == "Skylake":
            x = 1.5
            y = 0.65
        elif line_label == "BroadwellX":
            x = 3
            y = find_closest(items, 0.9)
        else:
            continue

        p = ax1.plot(cdf, marker='x', linewidth=2)
        ax1.annotate(line_label, xy=(0,0), xytext=(x, y), weight='light', color=p[0].get_color())

    plt.savefig(os.path.join(sys.argv[1], "IvyBridgeEP-CDF.png"), format='png', pad_inches=0.0)
    plt.savefig(os.path.join(sys.argv[1], "IvyBridgeEP-CDF.pdf"), format='pdf', pad_inches=0.0)
    plt.close()
