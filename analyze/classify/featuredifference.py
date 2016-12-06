#!/usr/bin/env python3

import os
import sys
import time
import argparse
import math
import logging
import pydotplus

import pandas as pd
import numpy as np
from numpy import linalg as lg

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import neural_network
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify import get_argument_parser
from analyze.classify.runtimes import get_runtime_dataframe, get_runtime_pivot_tables
from analyze.classify.generate_matrix import matrix_file_name
from analyze.classify.svm import make_svm_result_filename, make_weka_results_filename
from analyze.util import *

def get_data_set(args, pA, pB):
    X = []
    Y = []

    runtimes = get_runtime_dataframe(args.data_directory)
    for config, table in get_runtime_pivot_tables(runtimes):
        if config in args.config:
            for (A, values) in table.iterrows():
                for (i, normalized_runtime) in enumerate(values):
                    B = table.columns[i]

                    if A == pA and B == pB:
                        classification = True if normalized_runtime > args.cutoff else False
                        results_path = os.path.join(args.data_directory, config, "{}_vs_{}".format(A, B))
                        matrix_file_path = os.path.join(results_path, matrix_file_name(args.core, args.uncore, args.features))

                        if os.path.exists(os.path.join(results_path, 'completed')):
                            if not os.path.exists(matrix_file_path):
                                logging.error("No matrix file ({}) found, run the generate_matrix.py script first!".format(matrix_file_path))
                                sys.exit(1)

                            df = pd.read_csv(matrix_file_path, index_col=False)
                            logging.debug("Adding {} vs {} to training set class={} file={}".format(A, B, classification, matrix_file_path))
                            Y.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                            X.append(df)
                        else:
                            logging.info("Exclude unfinished directory {}".format(results_path))

    return (pd.concat(X), pd.concat(Y))

def dist(x,y):
    minlen = min(len(x), len(y))
    return np.sqrt(np.sum( (lg.norm(x[:minlen]) - lg.norm(y[:minlen]))**2 ))

def distvector(X, Y):
    div = []
    for col in X.columns:
        d = dist(X[col], Y[col])
        div.append( (d, col) )
    return sorted(div)


if __name__ == '__main__':
    parser = get_argument_parser('Produce decision trees for training data.')
    args = parser.parse_args()

    Xrs, _ = get_data_set(args, 'rdarray', 'stream')
    Xsr, _ = get_data_set(args, 'stream', 'rdarray')
    Xss, _ = get_data_set(args, 'stream', 'stream')
    Xrr, _ = get_data_set(args, 'rdarray', 'rdarray')

    diffXrsXsr = []
    for col in Xrs.columns:
        if col.startswith('AVG'):
            d = dist(Xrs[col], Xsr[col])
            diffXrsXsr.append( (d, col) )

    print("Xrs, Xsr")
    for d, name in distvector(Xrs, Xsr)[-10:]:
        print(d, name)

    print("Xrs, Xrr")
    for d, name in distvector(Xrs, Xrr)[-10:]:
        print(d, name)

    print("Xrs, Xss")
    for d, name in distvector(Xrs, Xss)[-10:]:
        print(d, name)
