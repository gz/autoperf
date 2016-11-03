#!/usr/bin/env python3

import os
import sys
import time
import argparse
import math
import subprocess
from multiprocessing import Pool, TimeoutError, cpu_count

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify import get_argument_parser

INPUT_FILES = [
    #"XY_training_without_AA700_training_L3-SMT_uncore_shared_paironly_125.csv",
    "XY_training_without_AA700_training_L3-SMT_uncore_shared_paironly_125_dropzero.csv",
    #"XY_training_without_BSCHOL_training_L3-SMT_uncore_shared_paironly_125.csv",
    "XY_training_without_BSCHOL_training_L3-SMT_uncore_shared_paironly_125_dropzero.csv",
    #"XY_training_without_CNEAL_training_L3-SMT_uncore_shared_paironly_125.csv",
    "XY_training_without_CNEAL_training_L3-SMT_uncore_shared_paironly_125_dropzero.csv",
    #"XY_training_without_FERR_training_L3-SMT_uncore_shared_paironly_125.csv",
    "XY_training_without_FERR_training_L3-SMT_uncore_shared_paironly_125_dropzero.csv",
    #"XY_training_without_HD1400_training_L3-SMT_uncore_shared_paironly_125.csv",
    "XY_training_without_HD1400_training_L3-SMT_uncore_shared_paironly_125_dropzero.csv",
    #"XY_training_without_NBODY_training_L3-SMT_uncore_shared_paironly_125.csv",
    "XY_training_without_NBODY_training_L3-SMT_uncore_shared_paironly_125_dropzero.csv",
    #"XY_training_without_PR700_training_L3-SMT_uncore_shared_paironly_125.csv",
    "XY_training_without_PR700_training_L3-SMT_uncore_shared_paironly_125_dropzero.csv",
    #"XY_training_without_SCLUS_training_L3-SMT_uncore_shared_paironly_125.csv",
    "XY_training_without_SCLUS_training_L3-SMT_uncore_shared_paironly_125_dropzero.csv",
    #"XY_training_without_SWAPT_training_L3-SMT_uncore_shared_paironly_125.csv",
    "XY_training_without_SWAPT_training_L3-SMT_uncore_shared_paironly_125_dropzero.csv",
    #"XY_training_without_TC1400_training_L3-SMT_uncore_shared_paironly_125.csv",
    "XY_training_without_TC1400_training_L3-SMT_uncore_shared_paironly_125_dropzero.csv"
]

CLASSPATH = [
    "/home/gz/Desktop/weka-3-8-0/weka.jar",
    "/home/gz/wekafiles/packages/LibSVM/lib/libsvm.jar",
    "/home/gz/wekafiles/packages/SVMAttributeEval/SVMAttributeEval.jar"
]

def weka_cmd_line(input_file, output_file):
    weka_args = 'weka.attributeSelection.SVMAttributeEval -s "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N 25" -X 1 -Y 0 -Z 0 -P 1.0E-25 -T 1.0E-10 -C 1.0 -N 0 -i'
    classpath = ':'.join(CLASSPATH)
    return "java -classpath {} {} {} > {}".format(classpath, weka_args, input_file, output_file)

def invoke_weka(input_file, output_file):
    java_cmd = weka_cmd_line(input_file, output_file)
    print ("About to execute", java_cmd)
    subprocess.call(java_cmd, shell=True)

if __name__ == '__main__':
    parser = get_argument_parser('Make weka ranking files.', arguments=['data'])
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help="Overwrite the file if it already exists.", default=False)
    args = parser.parse_args()

    pool = Pool(processes=4)
    results = []
    for ipf in INPUT_FILES:
        input_file = os.path.join(args.data_directory, ipf)
        output_file = os.path.join(args.data_directory, 'ranking_' + ipf.split('_', 1)[1])
        #invoke_weka(input_file, output_file)
        res = pool.apply_async(invoke_weka, (input_file, output_file))
        results.append(res)

    [r.get() for r in results]
    pool.close()
    pool.join()
