#!/usr/bin/env python3

import os
import sys
import time
import argparse
import math
import subprocess
import logging
from multiprocessing import Pool, TimeoutError, cpu_count

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify import get_argument_parser
from analyze.classify.svm import make_weka_results_filename
from analyze.classify.runtimes import get_runtime_dataframe
from analyze.classify.svm_topk import make_ranking_filename

CLASSPATH = [
    "/home/gz/Desktop/weka-3-8-0/weka.jar",
    "/home/gz/wekafiles/packages/LibSVM/lib/libsvm.jar",
    "/home/gz/wekafiles/packages/SVMAttributeEval/SVMAttributeEval.jar"
]

def weka_cmd_cfs(input_file, output_file):
    weka_args = 'weka.attributeSelection.CfsSubsetEval -s "weka.attributeSelection.GreedyStepwise -R -T -1.7976931348623157E308 -N 25 -num-slots 8" -P 8 -E 8 -i'
    classpath = ':'.join(CLASSPATH)
    return "java -classpath {} {} {} > {}".format(classpath, weka_args, input_file, output_file)

def weka_cmd_svmeval(input_file, output_file):
    weka_args = 'weka.attributeSelection.SVMAttributeEval -s "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N 25" -X 1 -Y 0 -Z 0 -P 1.0E-25 -T 1.0E-10 -C 1.0 -N 0 -i'
    classpath = ':'.join(CLASSPATH)
    return "java -classpath {} {} {} > {}".format(classpath, weka_args, input_file, output_file)

def invoke_weka(input_file, output_file, method):
    #java_cmd = weka_cmd_svmeval(input_file, output_file)
    if method == 'svm':
        java_cmd = weka_cmd_svmeval(input_file, output_file)
    elif method == 'cfs':
        java_cmd = weka_cmd_cfs(input_file, output_file)
    else:
        logging.error("Unknown method {}".format(method))
        sys.exit(1)
    print ("About to execute", java_cmd)
    subprocess.call(java_cmd, shell=True)

if __name__ == '__main__':
    parser = get_argument_parser('Make weka ranking files.')
    args = parser.parse_args()

    pool = Pool(processes=cpu_count())
    results = []
    runtimes = get_runtime_dataframe(args.data_directory)
    tests = [[x] for x in sorted(runtimes['A'].unique())]

    os.makedirs(os.path.join(args.data_directory, "ranking"), exist_ok=True)
    for test in tests:
        input_file = make_weka_results_filename('XY_training_without_{}'.format('_'.join(sorted(test))), args)
        input_path = os.path.join(args.data_directory, "matrices", input_file)

        method = 'svm'
        output_file = make_ranking_filename(test, args, method)
        output_path = os.path.join(args.data_directory, "ranking", output_file)
        res = pool.apply_async(invoke_weka, (input_path, output_path, method))
        results.append(res)

    [r.get() for r in results]
    pool.close()
    pool.join()
