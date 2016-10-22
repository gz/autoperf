#!/usr/bin/env python

import os
import sys
import pandas as pd
import numpy as np
if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..', ".."))
    from analyze import util


def persist_correlation(correlation_file, events, correlation_matrix):
    "Writes the correlation matrix (pairwise correlation value for every event) to file."
    with open(correlation_file, 'w') as f:
        header = '\t{}\n'.format('\t'.join([str(i) for i in events]))
        f.write(header)
        for i in events:
            f.write('{}'.format(i))
            for j in events:
                f.write('\t{}'.format(correlation_matrix.ix[i, j]))
            f.write('\n')

def persist_correlated_events(correlated_events_file, events, event_names, correlation_matrix):
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

def correlation_matrix(data_directory):
    df = load_as_X(os.path.join(data_directory, 'result.csv'))

    correlation_matrix = df.corr()
    # Ensure all values in correlation matrix are valid
    assert not correlation_matrix.isnull().values.any()

    correlation_file = os.path.join(data_directory, 'correlation_matrix.csv')
    correlation_matrix.to_csv(correlation_file)

    #from information_gain import ig
    #y = pd.DataFrame( [1 for _ in range(0, df.shape[0])] )
    #print (df.sum(axis=1).reshape(1, -1)) * y
    #print y.shape
    #print df.shape
    #print ig(df, y)

    sys.exit(1)

    # Get event names
    """
    events_file = os.path.join(data_directory, 'events.dat')
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
    """


def usage(progname):
    print >> sys.stderr, 'usage:', progname, '[data_input_dir]'
    sys.exit(0)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        usage(sys.argv[0])
    correlation_matrix(sys.argv[1])
