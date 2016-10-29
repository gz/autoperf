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

def extract_all(data_directory, uncore, overwrite):
    for root, dirs, files in os.walk(args.data_directory):
        if os.path.exists(os.path.join(root, 'completed')):
            print("Processing", root)

            if "uncore" in uncore:
                filename = os.path.join(root, "results_uncore_all.csv")
                if overwrite or not os.path.exists(filename):
                    subprocess.call([AUTOPERF_PATH, "extract", "-u", "all", "-o", filename, root])
                else:
                    print(("{} exists, skipping".format(filename)))

            if "shared" in uncore:
                filename = os.path.join(root, "results_uncore_shared.csv")
                if overwrite or not os.path.exists(filename):
                    subprocess.call([AUTOPERF_PATH, "extract", "-u", "shared", "-o", filename, root])
                else:
                    print(("{} exists, skipping".format(filename)))

            if "exclusive" in uncore:
                filename = os.path.join(root, "results_uncore_exclusive.csv")
                if overwrite or not os.path.exists(filename):
                    subprocess.call([AUTOPERF_PATH, "extract", "-u", "exclusive", "-o", filename, root])
                else:
                    print(("{} exists, skipping".format(filename)))

            if "none" in uncore:
                filename = os.path.join(root, "results_uncore_none.csv")
                if overwrite or not os.path.exists(filename):
                    subprocess.call([AUTOPERF_PATH, "extract", "-u", "none", "-o", filename, root])
                else:
                    print(("{} exists, skipping".format(filename)))
        else:
            print(("Exclude unfinished directory {}".format(root)))



if __name__ == '__main__':
    parser = get_argument_parser('Wrapper script for applying `autoperf extract` to all profiles in the data directory.', arguments=['data'])
    parser.add_argument('--uncore', dest='uncore', nargs='+', type=str, help="What uncore counters to include [all, shared, exclusive, none].", default=['shared'])
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help="Overwrite the file if it already exists.")
    args = parser.parse_args()

    if not os.path.exists(AUTOPERF_PATH) and not os.path.isfile(AUTOPERF_PATH):
        print("autoperf binary not found, do cargo build --release first!")
        sys.exit(2)

    extract_all(args.data_directory, args.uncore, args.overwrite)
