#!/usr/bin/env python3

import os
import sys

import pandas as pd

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify import get_argument_parser

if __name__ == '__main__':
    parser = get_argument_parser('Figures out how many events are 0.', arguments=['data', 'config', 'uncore'])
    #runtimes = get_runtime_dataframe(args.data_directory)
    X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args.data_directory, args.config, test, uncore=args.uncore, cutoff=args.cutoff, include_alone=args.include_alone)
