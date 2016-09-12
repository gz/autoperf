#!/bin/env python

import os
import sys

import pandas as pd
import numpy as np

from common import *
from config import *

def usage(progname):
    print >> sys.stderr, 'usage:', progname, '[plot_output_dir]'
    sys.exit(1)

def get_benchmark_name(benchmark):
    return '{}.{}'.format(
        '_'.join([p.name for p in benchmark.processes]),
        benchmark.scheme.name
    )

def get_process_name(benchmark, i):
    process_name = get_benchmark_name(benchmark)
    if benchmark.scheme is not Scheme.base:
        process_name += '.' + str(i + 1)
    return process_name

def get_data_file(plot_output_dir, benchmark, i):
    data_file = '{}.dat'.format(get_benchmark_name(benchmark))
    if benchmark.scheme is not Scheme.base:
        data_file += '.' + str(i + 1)
    data_file = os.path.join(plot_output_dir, data_file)
    return data_file

def persist_correlation(correlation_file, events, correlation_matrix):
    with open(correlation_file, 'w') as f:
        header = '\t{}\n'.format('\t'.join([str(i) for i in events]))
        f.write(header)
        for i in events:
            f.write('{}'.format(i))
            for j in events:
                f.write('\t{}'.format(correlation_matrix.ix[i, j]))
            f.write('\n')

def persist_correlated_events(correlated_events_file, events, event_names,
                              correlation_matrix):
    with open(correlated_events_file, 'w') as f:
        degree_to_events_dict = {}
        # Find correlated events for each event
        for i in events:
            correlated_events = [
                j for j in events
                if i != j and correlation_matrix.ix[i, j] >= CORRELATION_CUTOFF
            ]
            n = len(correlated_events)
            if n not in degree_to_events_dict:
                degree_to_events_dict[n] = []
            degree_to_events_dict[n].append(i)

            f.write('Event {} {} ({})\n'.format(i, event_names[i],
                                                len(correlated_events)))
            for j in correlated_events:
                f.write('\t{:3d} {:.2f} {}\n'.format(
                    j, correlation_matrix.ix[i, j], event_names[j])
                )
        f.write('-' * 50 + '\n')
        for n in sorted(degree_to_events_dict, reverse=True):
            l = degree_to_events_dict[n]
            f.write('{} ({}): {}\n'.format(n, len(l),
                                           ', '.join([str(i) for i in l])))

def persist_excluded_events(excluded_events_file, excluded_events):
    with open(excluded_events_file, 'w') as f:
        # Find correlated events for each event
        for i in excluded_events:
            f.write('{}\n'.format(i))

def get_benchmark_data_files(benchmark, plot_output_dir):
    verify_scheme(benchmark)
    data_files = []
    for process_idx, process in enumerate(benchmark.processes):
        process_name = get_process_name(benchmark, process_idx)
        data_file = get_data_file(plot_output_dir, benchmark, process_idx)
        data_files.append((process_name, data_file))
    return data_files

def main(argv):
    if len(argv) > 2:
        usage(argv[0])

    script_dir = os.path.dirname(os.path.realpath(__file__))
    if len(argv) > 1:
        plot_output_dir = argv[1]
    else:
        plot_output_dir = os.path.join(script_dir , '..', 'plot_output')

    # Get list of data files in format [(process, log), ...]
    data_files = []
    for i, benchmark in enumerate(BENCHMARKS):
        data_files.extend(get_benchmark_data_files(benchmark, plot_output_dir))

    # Read data files into DataFrame objects:
    # Column 0 is the time in ns
    # Column i, i > 0, is event i
    df_dict = {}
    for process, data_file in data_files:
        data = pd.read_csv(data_file, sep='\t', header=None, index_col=None,
                           comment='#')
        # First row consists only of 0's
        df_dict[process] = data.ix[1:]

    # Merge events of the same type into a vector
    df = pd.concat(df_dict)
    # Remove events whose deltas are all 0
    excluded_events = []
    for event in df.columns[1:]:
        if sum(df[event]) == 0:
            excluded_events.append(event)
            del df[event]

    # Get list of events after pruning
    events = list(df.columns[1:])
    correlation_matrix = pd.DataFrame(np.corrcoef([df[i] for i in events]),
                                      index=events, columns=events)
    assert correlation_matrix.shape == (len(events), len(events))

    # Ensure all values in correlation matrix are valid
    for i in events:
        for j in events:
            assert not np.isnan(correlation_matrix.ix[i, j])

    # Write correlation matrix
    correlation_file = os.path.join(plot_output_dir, 'event_correlation.dat')
    persist_correlation(correlation_file, events, correlation_matrix)

    # Get event names
    events_file = os.path.join(plot_output_dir, 'events.dat')
    event_names = pd.read_csv(events_file, sep='\t', header=None, index_col=0)
    event_names = event_names.to_dict()[1]

    # Write correlated events for each event
    correlated_events_file = os.path.join(
        plot_output_dir,
        'correlated_events.{}.dat'.format(int(CORRELATION_CUTOFF * 100))
    )
    persist_correlated_events(correlated_events_file, events, event_names,
                              correlation_matrix)

    # Write excluded events
    excluded_events_file = os.path.join(plot_output_dir,
                                        'excluded_events.dat')
    persist_excluded_events(excluded_events_file, excluded_events)

if __name__ == '__main__':
    main(sys.argv)
