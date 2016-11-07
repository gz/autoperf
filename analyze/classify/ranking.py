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

def weka_cmd_ig(input_file, output_file):
    weka_args = 'weka.classifiers.meta.AttributeSelectedClassifier -E "weka.attributeSelection.InfoGainAttributeEval " -S "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N 25"'
    classpath = ':'.join(CLASSPATH)
    return "java -classpath {} {} {} > {}".format(classpath, weka_args, input_file, output_file)

def weka_cmd_corr(input_file, output_file):
    weka_args = 'weka.filters.supervised.attribute.AttributeSelection -E "weka.attributeSelection.CorrelationAttributeEval " -S "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N 25"'
    classpath = ':'.join(CLASSPATH)
    return "java -classpath {} {} {} > {}".format(classpath, weka_args, input_file, output_file)

def invoke_weka(input_file, output_file, method):
    if method == 'svm':
        java_cmd = weka_cmd_svmeval(input_file, output_file)
    elif method == 'cfs':
        java_cmd = weka_cmd_cfs(input_file, output_file)
    elif method == 'ig':
        java_cmd = weka_cmd_if(input_file, output_file)
    else:
        logging.error("Unknown method {}".format(method))
        sys.exit(1)
    logging.debug("About to execute: {}".format(java_cmd))
    subprocess.call(java_cmd, shell=True)

if __name__ == '__main__':
    parser = get_argument_parser('Make weka ranking files.',
                                  arguments=['data', 'uncore', 'cutoff', 'config',
                                             'alone', 'features', 'dropzero',
                                             'ranking'])
    args = parser.parse_args()
    if args.ranking == 'svm':
        parallelism = cpu_count()
    else: # Rest should have built-in parallelization:
        parallelism = 1

    pool = Pool(processes=parallelism)
    results = []
    runtimes = get_runtime_dataframe(args.data_directory)
    tests = [[x] for x in sorted(runtimes['A'].unique())]

    os.makedirs(os.path.join(args.data_directory, "ranking"), exist_ok=True)
    for test in tests:
        input_file = make_weka_results_filename('XY_training_without_{}'.format('_'.join(sorted(test))), args)
        input_path = os.path.join(args.data_directory, "matrices", input_file)

        if not os.path.exists(input_path):
            logging.error("{} does not exist. Run the svm.py script with the --weka argument.".format(input_path))
            sys.exit(1)

        output_file = make_ranking_filename(test, args)
        output_path = os.path.join(args.data_directory, "ranking", output_file)
        res = pool.apply_async(invoke_weka, (input_path, output_path, args.ranking))
        results.append(res)

    [r.get() for r in results]
    pool.close()
    pool.join()
