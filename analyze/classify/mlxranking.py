import os
import sys
import multiprocessing

import pandas as pd

from sklearn import preprocessing
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..', ".."))
from analyze.classify import get_argument_parser
from analyze.classify.svm import make_weka_results_filename, CLASSIFIERS, row_training_and_test_set
from analyze.classify.runtimes import get_runtime_dataframe
from analyze.classify.svm_topk import make_ranking_filename

def generate_ranking(args):
    clf = CLASSIFIERS[args.kernel]
    if args.all:
        tests = [[None]]
    else:
        tests = [args.test]

    X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args, tests)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    if args.ranking == 'sfs':
        sfs = SFS(clf, k_features=25, forward=True, floating=False, scoring='accuracy', verbose=2, cv=4, n_jobs=multiprocessing.cpu_count()) # was 5
    elif args.ranking == 'sffs':
        sfs = SFS(clf, k_features=25, forward=True, floating=True, scoring='accuracy', verbose=2, n_jobs=5)
    sfs = sfs.fit(X_scaled, Y)

    df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T

    prev = set()
    events = []
    for idx, data in df.iterrows():
        cur = set(data['feature_idx'])
        new = cur - prev
        if len(new) == 1:
            events.append(X.columns[new.pop()])
        else:
            events.append("None ({})".format(len(new)))
        prev = cur

    series = pd.Series(events)
    df.reset_index(inplace=True)
    df['name'] = series

    filename = make_ranking_filename([args.test], args)
    df.to_csv(os.path.join(output_directory, filename), index=False)
    return df

if __name__ == '__main__':
    parser = get_argument_parser('Feature selection for the attributes.',
                arguments=['data', 'core', 'uncore', 'cutoff', 'config', 'alone', 'features', 'dropzero', 'ranking'])
    parser.add_argument('--kernel', dest='kernel', type=str, help="Which kernel to use for ranking.", required=True)
    parser.add_argument('--test', dest='test', type=str, help="Which app to generate the ranking for.", required=False)
    parser.add_argument('--all', dest='all', action='store_true', default=False, help="Use whole data set as training set.")

    args = parser.parse_args()
    assert args.ranking == "sfs" or args.ranking == "sffs"

    runtimes = get_runtime_dataframe(args.data_directory)

    output_directory = os.path.join(args.data_directory, "ranking")
    os.makedirs(output_directory, exist_ok=True)

    generate_ranking(args)
