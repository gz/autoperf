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

def generate_ranking(args, test):
    clf = CLASSIFIERS["poly2balancedC2"]
    X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args, test)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    sfs = SFS(clf, k_features=25, forward=True, floating=False, scoring='accuracy', verbose=2,
              cv=3, n_jobs=1)
    sfs = sfs.fit(X_scaled, Y)

    df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T

    prev = set()
    events = []
    for idx, data in df.iterrows():
        cur = set(data['feature_idx'])
        new = cur - prev
        assert len(new) == 1
        events.append(X.columns[new.pop()])
        prev = cur

    series = pd.Series(events)
    df.reset_index(inplace=True)
    df['name'] = series

    filename = make_ranking_filename(test, args)
    df.to_csv(os.path.join(output_directory, filename), index=False)
    return df

if __name__ == '__main__':
    parser = get_argument_parser('Feature selection for the attributes.',
                arguments=['data', 'uncore', 'cutoff', 'config', 'alone', 'features', 'dropzero', 'ranking'])
    args = parser.parse_args()
    assert args.ranking == "sfs"

    runtimes = get_runtime_dataframe(args.data_directory)
    tests = [[x] for x in sorted(runtimes['A'].unique())]

    output_directory = os.path.join(args.data_directory, "ranking")
    os.makedirs(output_directory, exist_ok=True)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    rows = []

    for test in tests:
        res = pool.apply_async(generate_ranking, (args, test))
        rows.append(res)

    results_table = pd.concat([r.get() for r in rows], ignore_index=True)
    pool.close()
    pool.join()
