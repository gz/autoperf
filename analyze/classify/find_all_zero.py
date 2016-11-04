#!/usr/bin/env python3
import os
import sys

import pandas as pd

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify import get_argument_parser
from analyze.classify import svm
from analyze.util import get_zero_features_in_matrix

def mkfilename(prefix, configs, uncore, features):
    OUT_FILE = "{}_uncore_{}_features_{}.csv"
    return OUT_FILE.format(prefix, uncore, '_'.join(sorted(features)))

def calculate_zero_features(data_directory, configs, uncore, features):
    # drop_zero has to be false, otherwise we have recursion bug
    X, Y, Y_weights, X_test, Y_test = svm.row_training_and_test_set(data_directory, configs, [None], \
        uncore=uncore, features=features, cutoff=1.00, \
        include_alone=True, drop_zero=False)
    features = get_zero_features_in_matrix(X)
    return features

def zero_features(data_directory, configs, uncore, features, overwrite):
    feature_filename = mkfilename("zero_features", configs, uncore, features)
    zero_features_path = os.path.join(data_directory, feature_filename)
    event_filename = mkfilename("zero_events", configs, uncore, features)
    zero_events_path = os.path.join(data_directory, event_filename)

    if not os.path.exists(zero_features_path) or overwrite:
        features = calculate_zero_features(data_directory, configs, uncore, features)
        df = pd.DataFrame(features)
        df.to_csv(zero_features_path, index=False, header=['EVENT_NAME'])

        avg_features = filter(lambda x: x.startswith("AVG."), features)
        events = sorted(set([ feature.split(".", 1)[1] for feature in avg_features ]))
        df = pd.DataFrame(events)
        df.to_csv(zero_events_path, index=False, header=['EVENT_NAME'])

    assert os.path.exists(zero_features_path)
    return pd.read_csv(zero_features_path)

if __name__ == '__main__':
    parser = get_argument_parser('Figures out what events are always 0.',
                                 arguments=['data', 'config', 'uncore', 'features', 'overwrite'])
    args = parser.parse_args()
    print(zero_features(args.data_directory, args.config, args.uncore, args.features, args.overwrite))
