#!/usr/bin/env python3
import os
import sys
import time
import argparse

from multiprocessing import Pool, TimeoutError
import pandas as pd
import numpy as np

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.classify import get_argument_parser_uncore
from analyze.util import *


def make_matrix(results_file, output_file, aggregations):
    print("Processing {}".format(results_file))
    df = load_as_X(results_file, aggregate_samples=aggregations, cut_off_nan=True)
    print("SAVING {}".format(output_file))
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    parser = get_argument_parser_uncore('Generates matrix files for use with ML algorithms.')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help="Overwrite the file if it already exists.", default=False)
    parser.add_argument('--aggregation', dest='aggregations', nargs='+', type=str,
                        help="What uncore counters to include (default mean std min max).",
                        default=['mean', 'std', 'min', 'max'],
                        choices=['mean', 'std', 'min', 'max'])
    args = parser.parse_args()

    ## Settings:
    INPUT_RESULTS_FILE = 'results_uncore_{}.csv'
    OUT_FILE = 'matrix_X_uncore_{}_aggregation_{}.csv'

    pool = Pool(processes=1)
    num = 0
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

        results_file = os.path.join(results_path, INPUT_RESULTS_FILE)
        output_file = os.path.join(results_path, OUT_FILE)

        if os.path.exists(os.path.join(results_path, 'completed')):
            for uncore in args.uncore:
                results_file = os.path.join(results_path, INPUT_RESULTS_FILE.format(uncore))
                output_file = os.path.join(results_path, OUT_FILE.format(uncore, args.aggregations))

                if not os.path.exists(output_file) or args.overwrite:
                    print(("Processing {} vs. {} ({})".format(A, B, uncore)))
                    pool.apply_async(make_matrix, (results_file, output_file, args.aggregations))
                else:
                    print(("{} already exists, skipping.".format(output_file)))
        else:
            print(("Exclude unfinished directory {}".format(results_path)))

    pool.close()
    pool.join()
