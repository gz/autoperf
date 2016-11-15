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


if __name__ == '__main__':
    parser = get_argument_parser('Feature selection for the attributes.',
                arguments=['data', 'uncore', 'cutoff', 'config', 'alone', 'features', 'dropzero', 'ranking'])
    args = parser.parse_args()
    assert args.ranking == "sfs"

    runtimes = get_runtime_dataframe(args.data_directory)
    tests = [[x] for x in sorted(runtimes['A'].unique())]

    clf = CLASSIFIERS["poly2balancedC1.5"]

    output_directory = os.path.join(args.data_directory, "ranking")
    os.makedirs(output_directory, exist_ok=True)

    for test in tests:
        X, Y, Y_weights, X_test, Y_test = row_training_and_test_set(args, test)

        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        sfs = SFS(clf, k_features=25, forward=True, floating=False, scoring='accuracy', verbose=2,
                  cv=3, n_jobs=int(multiprocessing.cpu_count() / 2))
        sfs = sfs.fit(X_scaled, Y)

        df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
        filename = make_ranking_filename(test, args)
        df.to_csv(os.path.join(output_directory, filename + ".csv"), index=False)
        print(df)
