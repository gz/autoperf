#!/usr/bin/env python3
"""
Compares two timeseries and looks for differences.

It does that by summing up the maximas observed for every event on a given
slice of the time-series. Which leads to a single value (magnitude) per
observed event.
Then given two timeseries A and B we compare them by calculating a normalized
factor A.event / (A.event + B.event) to find values that predominantly trigger
only in A and B.event / (A.event + B.event) to find values that trigger
predominantly in B.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib


def timeseries_file(data_directory):
    timeseries_file = os.path.join(data_directory, 'timeseries_avg_nonzero.csv')
    if os.path.exists(timeseries_file):
        return pd.read_csv(timeseries_file, index_col=0, skipinitialspace=True)
    else:
        print("Generating timeseries_avg_nonzero.csv")
        timeseries = util.load_as_X(os.path.join(data_directory, 'results.csv'),
                aggregate_samples=['mean'], cut_off_nan=True, remove_zero=True)
        timeseries.to_csv(timeseries_file)
        return timeseries

def usage(progname):
    print('usage:', progname, '[data_input_dir for A] [data_input_dir for B]')
    sys.exit(0)

if __name__ == '__main__':
    sys.path.insert(1, os.path.join(sys.path[0], '..', ".."))
    from analyze import util

    if len(sys.argv) > 3:
        usage(sys.argv[0])

    dfA = timeseries_file(sys.argv[1])
    dfA = dfA[-15:].sum() # TODO range is hard-coded, adjust
    
    dfB = timeseries_file(sys.argv[2])
    dfB = dfB[-15:].sum() # TODO range is hard-coded, adjust

    max_among_both = pd.concat([dfA, dfB]).max(level=0)

    normA = (dfA / (dfA + dfB)).dropna()
    normB = (dfB / (dfA + dfB)).dropna()

    fmt_string = "{event}: {fraction:.2f} ({absolute1}-{absolute2}={res})"


    print("Events that predominantly trigger in {} and not in {}\n".format(sys.argv[2], sys.argv[1]))
    print("Event name: Fraction (progB - progA = difference)")
    print("=================================================")
    for (name, val) in normB.sort_values().iteritems():
        if val > 0.95:
            print (fmt_string.format(event=name, fraction=val, absolute1=dfB[name], absolute2=dfA[name], res=dfB[name]-dfA[name]))
    
    print("")
    print("")

    print("Events that predominantly trigger in {} and not in {}\n".format(sys.argv[1], sys.argv[2]))
    
    print("Event name: Fraction (progA - progB = difference)")
    print("=================================================")
    for (name, val) in normA.sort_values().iteritems():
        if val > 0.95:
            print (fmt_string.format(event=name, fraction=val, absolute1=dfA[name], absolute2=dfB[name], res=dfA[name]-dfB[name]))

