import os, sys

from matplotlib import pyplot as plt, font_manager
import numpy as np
import pandas as pd

ticks_font = font_manager.FontProperties(family='Decima Mono')
plt.style.use([os.path.join(sys.path[0], 'ethplot.mplstyle')])
NAME = "counters_vs_events"

if __name__ == '__main__':
    fig = plt.figure()

    raw_data = pd.read_csv(os.path.join(sys.argv[1], "events.csv"), skipinitialspace=True)
    raw_data['year'].map(lambda y: int(y))
    raw_data.sort_values(by=['year', 'counters'], inplace=True)
    raw_data.set_index('year', inplace=True)
    raw_data['events'] = raw_data['core events'] + raw_data['uncore events']
    print raw_data
    #raw_data.plot()
    #ycounters = raw_data['counters'] #.drop_duplicates(keep='last')

    LEFT = -0.056
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.set_ylabel('Hardware Performance Events', rotation='horizontal', horizontalalignment='left')
    ax1.set_xlabel('Year of Release')

    ax1.yaxis.set_label_coords(LEFT-0.030, 1.03)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_xlim(2007.5, 2016.5)

    for row in raw_data.iterrows():
        year = row[0]
        name = row[1]['architecture']
        events = row[1]['events']

        if year == 2016:
            ax1.annotate(name, xy=(year, events), xytext=(2015.2, ytext), weight='light')
        else:
            ytext = events
            if name == "Westmere EP":
                ytext -= 20
            ax1.annotate(name, xy=(year, events), xytext=(year+0.1, ytext), weight='light')

    p = ax1.plot(raw_data['events'], marker='o', linestyle='None')
    p = ax1.plot(raw_data['counters'])

    ax1.annotate("HW Counters per Core", xy=(2014, 0), xytext=(2013.4, 29), color=p[0].get_color())

    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.setp(ax1.get_xticklabels(), fontproperties=ticks_font)
    plt.setp(ax1.get_yticklabels(), fontproperties=ticks_font)

    plt.savefig(NAME + ".pdf", format='pdf')
