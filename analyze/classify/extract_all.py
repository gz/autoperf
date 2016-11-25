#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calls extract on all result folders.
"""

import sys
import os
import subprocess
import argparse

AUTOPERF_PATH = os.path.join(sys.path[0], "..", "..", "target", "release", "autoperf")

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify import get_argument_parser

def extract_all(data_directory, core, uncore, overwrite):
    for root, dirs, files in os.walk(args.data_directory):
        if os.path.exists(os.path.join(root, 'completed')):
            print("Processing", root)
            filename = os.path.join(root, "results_core_{}_uncore_{}.csv".format(core, uncore))
            if overwrite or not os.path.exists(filename):
                subprocess.call([AUTOPERF_PATH, "extract", "-c", core, "-u", uncore, "-o", filename, root])
            else:
                print(("{} exists, skipping".format(filename)))

        else:
            print(("Exclude unfinished directory {}".format(root)))

if __name__ == '__main__':
    parser = get_argument_parser('Wrapper script for applying `autoperf extract` to all profiles in the data directory.', arguments=['data', 'overwrite', 'uncore', 'core'])

    args = parser.parse_args()

    if not os.path.exists(AUTOPERF_PATH) and not os.path.isfile(AUTOPERF_PATH):
        print("autoperf binary not found, do cargo build --release first!")
        sys.exit(2)

    extract_all(args.data_directory, args.core, args.uncore, args.overwrite)
