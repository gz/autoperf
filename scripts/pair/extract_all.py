#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calls extract on all result folders.
"""

import sys
import os
import subprocess

AUTOPERF_PATH = os.path.join(sys.path[0], "..", "..", "target", "release", "autoperf")

def extract_all(data_directory):
    for root, dirs, files in os.walk(sys.argv[1]):
        if os.path.exists(os.path.join(root, 'completed')):
            print "Processing", root
            filename = os.path.join(root, "results_uncore_all.csv")
            subprocess.call([AUTOPERF_PATH, "extract", "-u", "all", "-o", filename, root])

            filename = os.path.join(root, "results_uncore_shared.csv")
            subprocess.call([AUTOPERF_PATH, "extract", "-u", "shared", "-o", filename, root])

            filename = os.path.join(root, "results_uncore_exclusive.csv")
            subprocess.call([AUTOPERF_PATH, "extract", "-u", "exclusive", "-o", filename, root])

            filename = os.path.join(root, "results_uncore_none.csv")
            subprocess.call([AUTOPERF_PATH, "extract", "-u", "none", "-o", filename, root])
        else:
            print "Exclude unfinished directory {}".format(root)



if __name__ == '__main__':
    if len(sys.argv) >= 2:
        data_directory = sys.argv[1]
    else:
        print "Usage: %s <data directory>" % sys.argv[0]
        sys.exit(1)

    if not os.path.exists(AUTOPERF_PATH) and not os.path.isfile(AUTOPERF_PATH):
        print "autoperf binary not found, do cargo build --release first!"
        sys.exit(2)

    extract_all(data_directory)
