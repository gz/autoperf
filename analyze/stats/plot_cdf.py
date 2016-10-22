import os, sys
import glob

from matplotlib import pyplot as plt, font_manager
import numpy as np
import pandas as pd

ticks_font = font_manager.FontProperties(family='Decima Mono')
plt.style.use([os.path.join(sys.path[0], '../ethplot.mplstyle')])

if __name__ == '__main__':

    for f in glob.glob(os.path.join(sys.argv[1], 'editdist_*.csv')):
        filename, file_extension = os.path.splitext(os.path.basename(f))

        raw_data = pd.read_csv(f, skipinitialspace=True)
        series = raw_data['edit distance']
        series = series.sort_values()
        series[len(series)] = series.iloc[-1]
        cum_dist = np.linspace(0.,1.,len(series))
        cdf = pd.Series(cum_dist, index=series)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_ylabel('CDF [%]')
        ax1.set_xlabel('Edit distance')

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.get_xaxis().tick_bottom()
        ax1.get_yaxis().tick_left()

        p = ax1.plot(cdf)

        ax1.get_xaxis().get_major_formatter().set_useOffset(False)
        plt.setp(ax1.get_xticklabels(), fontproperties=ticks_font)
        plt.setp(ax1.get_yticklabels(), fontproperties=ticks_font)

        plt.savefig(os.path.join(sys.argv[1], filename + ".png"), format='png', pad_inches=0.0)
        plt.savefig(os.path.join(sys.argv[1], filename + ".pdf"), format='pdf', pad_inches=0.0)
