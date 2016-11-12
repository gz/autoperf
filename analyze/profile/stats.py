#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gathers and prints some information gathered from the CSV file.
Mostly intended for sanity cecking only.
"""

import sys, os
import pandas as pd

from ascii_graph import Pyasciigraph

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.util import get_all_zero_events

def histogram(L):
    d = {}
    for x in L:
        if x in d:
            d[x] += 1
        else:
            d[x] = 1
    return d


def yield_cpu_sample_lengths(df):
    for idx in df.index.unique():
        if not idx.startswith("uncore_"):
            yield len(df.loc[[idx], 'SAMPLE_VALUE'])

def yield_uncore_sample_lengths(df):
    for idx in df.index.unique():
        if idx.startswith("uncore_"):
            yield len(df.loc[[idx], 'SAMPLE_VALUE'])

def samples_histogram(df, lengths_fn):
    lengths = histogram(lengths_fn(df))
    data = []
    for key, value in lengths.items():
        data.append( ("%d samples" % key, value) )
    data = sorted(data, key=lambda x: x[1])
    return data

if __name__ == '__main__':
    data_directory = sys.argv[1]
    df = pd.read_csv(os.path.join(data_directory, 'result.csv'), index_col=0, skipinitialspace=True)

    all_events = df.index.unique()
    all_zero = get_all_zero_events(df)

    print("Total Events measured:", len(all_events))
    title = "List of event samples that reported only zeroes (%d / %d):" % (len(all_zero), len(all_events))
    print('\n  - '.join([title] + all_zero))
    df = df.drop(all_zero)

    # Sample histogram
    graph = Pyasciigraph()
    for line in graph.graph('Recorded CPU samples histogram:', samples_histogram(df, yield_cpu_sample_lengths)):
        print(line.encode('utf-8'))

    for line in graph.graph('Recorded uncore samples histogram:', samples_histogram(df, yield_uncore_sample_lengths)):
        print(line.encode('utf-8'))

    # TODO: Should be CPU events
    print("The five events with fewest samples are:")
    for idx in sorted(df.index.unique(), key=lambda x: len(df.loc[[x], 'SAMPLE_VALUE']))[:100]:
        print(idx, ":", len(df.loc[[idx], 'SAMPLE_VALUE']), "samples")
