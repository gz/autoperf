#!/usr/bin/env python
import os
import sys
import time
import argparse

from multiprocessing import Pool, TimeoutError
import pandas as pd
import numpy as np

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.util import *

def make_matrix(results_file, output_file):
    print("Processing {}".format(results_file))
    df = load_as_X(results_file, aggregate_samples=['mean', 'std', 'min', 'max'], cut_off_nan=True)
    print("SAVING {}".format(output_file))
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 5)
    pd.set_option('display.width', 160)

    parser = argparse.ArgumentParser(description='Generates X and Y matrix files for use with ML algorithms.')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help="Overwrite the file if it already exists.", default=False)
    parser.add_argument('--uncore', dest='uncore', nargs='+', type=str, help="What uncore counters to include.", default=['shared'], choices=['all', 'shared', 'exclusive', 'none'])
    parser.add_argument('data_directory', type=str, help="Data directory root.")
    args = parser.parse_args()


    ## Settings:
    RESULTS_FILE = 'results_uncore_{}.csv'
    OUT_FILE = 'matrix_X_uncore_{}.csv'

    pool = Pool(processes=4)
    num = 0
    runtimes = get_runtime_dataframe(args.data_directory)
    for row in runtimes.itertuples():
        A = row.A
        B = row.B
        config = row.config
        normalized_runtime = row[4]
        results_path = None
        if pd.isnull(B) or B == "Alone":
            results_path = os.path.join(sys.argv[1], config, "{}".format(A))
        else:
            results_path = os.path.join(sys.argv[1], config, "{}_vs_{}".format(A, B))

        results_file = os.path.join(results_path, RESULTS_FILE)
        output_file = os.path.join(results_path, OUT_FILE)

        if os.path.exists(os.path.join(results_path, 'completed')):
            for uncore in args.uncore:
                results_file = os.path.join(results_path, RESULTS_FILE.format(uncore))
                output_file = os.path.join(results_path, OUT_FILE.format(uncore))

                if not os.path.exists(output_file) or args.overwrite:
                    print(("Processing {} vs. {} ({})".format(A, B, uncore)))
                    pool.apply_async(make_matrix, (results_file, output_file))
                else:
                    print(("{} already exists, skipping.".format(output_file)))
        else:
            print(("Exclude unfinished directory {}".format(results_path)))

    pool.close()
    pool.join()
