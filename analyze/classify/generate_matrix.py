#!/usr/bin/env python3
import os
import sys
import time
import argparse

from multiprocessing import Pool, TimeoutError, cpu_count
import pandas as pd
import numpy as np

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.classify import get_argument_parser
from analyze.util import *

def make_matrix(input_file, output_file, features):
    print("Processing {}".format(input_file))
    df = load_as_X(input_file, aggregate_samples=features, cut_off_nan=True)
    print("Saving {}".format(output_file))
    df.to_csv(output_file, index=False)

def matrix_file_name(core, uncore, features):
    OUT_FILE = 'matrix_X_core_{}_uncore_{}_features_{}.csv'
    return OUT_FILE.format(core, uncore, '_'.join(sorted(features)))

if __name__ == '__main__':
    parser = get_argument_parser('Generates matrix files for use with ML algorithms.', arguments=['data', 'uncore', 'core', 'features', 'overwrite'])
    args = parser.parse_args()

    ## Settings:
    INPUT_RESULTS_FILE = 'results_core_{}_uncore_{}.csv'

    pool = Pool(processes=cpu_count())
    async_results = []
    runtimes = get_runtime_dataframe(args.data_directory)
    for row in runtimes.itertuples():
        A = row.A
        B = row.B
        config = row.config
        normalized_runtime = row[4]
        results_path = None
        if pd.isnull(B) or B == "Alone":
            results_path = os.path.join(args.data_directory, config, "{}".format(A))
        else:
            results_path = os.path.join(args.data_directory, config, "{}_vs_{}".format(A, B))

        if os.path.exists(os.path.join(results_path, 'completed')):
            input_file = os.path.join(results_path, INPUT_RESULTS_FILE.format(args.core, args.uncore))
            output_file = os.path.join(results_path, matrix_file_name(args.core, args.uncore, args.features))

            if not os.path.exists(output_file) or args.overwrite:
                print("Processing {} vs. {} ({})".format(A, B, args.uncore))
                res = pool.apply_async(make_matrix, (input_file, output_file, args.features))
                async_results.append(res)
                #make_matrix(input_file, output_file, args.features)
            else:
                print(("{} already exists, skipping.".format(output_file)))
        else:
            print(("Exclude unfinished directory {}".format(results_path)))

    [r.get() for r in async_results]
    pool.close()
    pool.join()
