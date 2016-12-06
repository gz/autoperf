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

CLASSIFIERS = {
    'decision': tree.DecisionTreeClassifier(presort=True),
    #'decision5': tree.DecisionTreeClassifier(max_features=5),
    #'decision10': tree.DecisionTreeClassifier(max_features=10),
}

CLASSIFICATION = {
    'rdarraystream': 1,
    'rdarrayrdarray': 0,
    'streamrdarray': 1,
    'streamstream': 0,
}

def get_data_set(args, exclude):
    X = []
    Y = []

    runtimes = get_runtime_dataframe(args.data_directory)
    for config, table in get_runtime_pivot_tables(runtimes):
        if config in args.config:
            for (A, values) in table.iterrows():
                for (i, normalized_runtime) in enumerate(values):
                    B = table.columns[i]

                    classification = True if A == B else False
                    if B == "Alone" or B == None:
                        if not args.include_alone:
                            logging.debug("Skipping the samples with {} alone".format(A))
                            continue
                        results_path = os.path.join(args.data_directory, config, "{}".format(A))
                    else:
                        results_path = os.path.join(args.data_directory, config, "{}_vs_{}".format(A, B))
                    matrix_file_path = os.path.join(results_path, matrix_file_name(args.core, args.uncore, args.features))

                    if os.path.exists(os.path.join(results_path, 'completed')):
                        if not os.path.exists(matrix_file_path):
                            logging.error("No matrix file ({}) found, run the generate_matrix.py script first!".format(matrix_file_path))
                            sys.exit(1)

                        df = pd.read_csv(matrix_file_path, index_col=False)
                        if not (A, B) in exclude:
                            logging.debug("Adding {} vs {} to training set class={} file={}".format(A, B, classification, matrix_file_path))
                            Y.append(pd.Series([classification for _ in range(0, df.shape[0])]))
                            X.append(df)
                        else:
                            logging.debug("Skipping {}, because it was excluded.".format(results_path))
                    else:
                        logging.info("Exclude unfinished directory {}".format(results_path))

    return (pd.concat(X), pd.concat(Y))


if __name__ == '__main__':
    parser = get_argument_parser('Produce decision trees for training data.')
    parser.add_argument('--exclude', dest='exclude', nargs='+', type=str, help="List or programs to exclude from the data-set for the tree.")
    args = parser.parse_args()

    if not args.exclude:
        runtimes = get_runtime_dataframe(args.data_directory)
        exclude = []
    else:
        logging.debug("Exclude: {}".format(args.exclude))
        exclude = [ tuple(pair.split(',')) for pair in args.exclude ]

    output_directory = os.path.join(args.data_directory, "decision_trees")
    os.makedirs(output_directory, exist_ok=True)
    for kconfig, clf in list(CLASSIFIERS.items()):
        print("Trying kernel {}".format(kconfig))
        results_table = pd.DataFrame()

        logging.debug("Exclude: {}".format(exclude))
        X, Y = get_data_set(args, exclude)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        clf.fit(X_scaled, Y)

        #clf.fit(X, Y)

        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X.columns)
        filename = make_svm_result_filename("decision_tree", args, kconfig)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(os.path.join(output_directory, filename + ".pdf"))

        X['Y'] = Y
        X['Y'] = X['Y'].map(lambda x: 'Y' if x else 'N')
        exclude_strings = [ "{}vs{}".format(A, B) for A, B in exclude ]
        training_file_name = make_weka_results_filename('XY_without_{}'.format('_'.join(sorted(exclude_strings))), args)
        X.to_csv(os.path.join(output_directory, training_file_name), index=False)
