#!/usr/bin/env python

import os
import sys
import time

from multiprocessing import Pool, TimeoutError
import pandas as pd
import numpy as np

from runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from util import *

def add_to_classifier(X, Y):
    X = None
    Y = None
    return (X, Y)

def make_matrix(results_file, output_file):
    df = load_as_X(results_file, aggregate_samples='meanstd', cut_off_nan=True)
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 5)
    pd.set_option('display.width', 160)

    ## Settings:
    PARALLELISM = 6
    RESULTS_FILE = 'results_uncore_shared.csv'
    OUT_FILE = 'matrix_X_uncore_shared.csv'
    TO_BUILD = ['L3-SMT'] # 'L3-SMT-cores'
    CLASSIFIER_CUTOFF = 1.15

    pool = Pool(processes=PARALLELISM)
    num = 0
    runtimes = get_runtime_dataframe(sys.argv[0])
    for config, table in get_runtime_pivot_tables(runtimes):
        if config in TO_BUILD:
            for (A, values) in table.iterrows():
                for (i, normalized_runtime) in enumerate(values):
                    B = table.columns[i]
                    #print "{} with {} has interference {}".format(A, B, normalized_runtime > CLASSIFIER_CUTOFF)
                    results_path = os.path.join(sys.argv[1], config, "{}_vs_{}".format(A, B))
                    results_file = os.path.join(results_path, RESULTS_FILE)
                    output_file = os.path.join(results_path, OUT_FILE)

                    if os.path.exists(os.path.join(results_path, 'completed')):
                        if not os.path.exists(output_file):
                            print "Processing {} vs. {}".format(A, B)
                            pool.apply_async(make_matrix, (results_file, output_file))
                        else:
                            print "{} already exists, skipping.".format(output_file)
                    else:
                        print "Exclude unfinished directory {}".format(results_path)

    pool.close()
    pool.join()
