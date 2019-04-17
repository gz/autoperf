#!/usr/bin/env python3
"""
Compute the timeseries data as a CSV file.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib


if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], '..', ".."))
    from analyze import util

def timeseries_file(data_directory):
    timeseries = util.load_as_X(os.path.join(data_directory, 'results.csv'), aggregate_samples = ['mean', 'std', 'max', 'min'], cut_off_nan=True)
    timeseries_file = os.path.join(data_directory, 'timeseries.csv')
    timeseries.to_csv(timeseries_file)
    print("Generated timeseries.csv")

def usage(progname):
    print('usage:', progname, '[data_input_dir]')
    sys.exit(0)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        usage(sys.argv[0])
    timeseries_file(sys.argv[1])

