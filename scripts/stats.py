#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gathers and prints some information gathered from the CSV file.
Mostly intended for sanity cecking only.
"""

import sys, os
import pandas as pd

from util import get_all_zero_events
from ascii_graph import Pyasciigraph

def histogram(L):
    d = {}
    for x in L:
        if x in d:
            d[x] += 1
        else:
            d[x] = 1
    return d


def yield_sample_lengths(df):
    for idx in df.index.unique():
        yield len(df.loc[[idx], 'SAMPLE_VALUE'])

if __name__ == '__main__':
    data_directory = sys.argv[1]
    df = pd.read_csv(os.path.join(data_directory, 'result.csv'), index_col=0, skipinitialspace=True)

    all_events = df.index.unique()
    all_zero = get_all_zero_events(df)

    print "Total Events:", len(all_events)
    title = "Event samples reported all zeroes (%d / %d):" % (len(all_zero), len(all_events))
    print '\n  - '.join([title] + all_zero)
    df = df.drop(all_zero)

    # Sample histogram
    graph = Pyasciigraph()
    lengths = histogram(yield_sample_lengths(df))
    data = []
    for key, value in lengths.iteritems():
        data.append( ("%d samples" % key, value) )
    data.sort()

    for line in graph.graph('Recorded samples histogram:', data):
        print line.encode('utf-8')

    for idx in df.index.unique():
        print idx, len(df.loc[[idx], 'SAMPLE_VALUE'])
