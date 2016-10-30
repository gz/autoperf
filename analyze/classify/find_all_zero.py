#!/usr/bin/env python3

import os
import sys

import pandas as pd

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify import get_argument_parser
from analyze.classify import svm
from analyze.util import get_zero_features_in_matrix

def mkfilename(prefix, configs, uncore):
    return "{}_{}_uncore_{}.csv".format(prefix, '_'.join(configs), uncore)

def all_zero_features(configs, uncore):
    feature_filename = mkfilename("zero_features", args.config, args.uncore)
    df.to_csv(os.path.join(args.data_directory, feature_filename), index=False)

if __name__ == '__main__':
    parser = get_argument_parser('Figures out how many events are 0.', arguments=['data', 'config', 'uncore'])
    args = parser.parse_args()

    X, Y, Y_weights, X_test, Y_test = svm.row_training_and_test_set(args.data_directory, args.config, [None], uncore=args.uncore, cutoff=1.00, include_alone=False)

    features = get_zero_features_in_matrix(X)
    df = pd.DataFrame(features)
    feature_filename = mkfilename("zero_features", args.config, args.uncore)
    df.to_csv(os.path.join(args.data_directory, feature_filename), index=False, header=['EVENT_NAME'])

    events = sorted(set([ feature.split(".", 1)[1] for feature in features ]))
    event_filename = mkfilename("zero_events", args.config, args.uncore)
    df = pd.DataFrame(events)
    df.to_csv(os.path.join(args.data_directory, event_filename), index=False, header=['EVENT_NAME'])
