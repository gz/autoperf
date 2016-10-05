#!/usr/bin/env python

import os
import sys
import pandas as pd
import numpy as np

from util import *

from sklearn import tree

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import pydotplus

def information_gain(x, y):

    def _entropy(values):
        counts = np.bincount(values)
        probs = counts[np.nonzero(counts)] / float(len(values))
        return - np.sum(probs * np.log(probs))

    def _information_gain(feature, y):
        print np.nonzero(feature)
        feature_set_indices = np.nonzero(feature)[1]
        feature_not_set_indices = [i for i in feature_range if i not in feature_set_indices]
        entropy_x_set = _entropy(y[feature_set_indices])
        entropy_x_not_set = _entropy(y[feature_not_set_indices])

        return entropy_before - (((len(feature_set_indices) / float(feature_size)) * entropy_x_set)
                                 + ((len(feature_not_set_indices) / float(feature_size)) * entropy_x_not_set))

    feature_size = x.shape[0]
    feature_range = range(0, feature_size)
    entropy_before = _entropy(y)
    information_gain_scores = []

    for feature in x.T:
        information_gain_scores.append(_information_gain(feature, y))
    return information_gain_scores, []

if __name__ == '__main__':
    raw_data1 = pd.read_csv(os.path.join(sys.argv[1], 'transformed.csv'), sep=',', skipinitialspace=True, index_col=0)
    raw_data2 = pd.read_csv(os.path.join(sys.argv[2], 'transformed.csv'), sep=',', skipinitialspace=True, index_col=0)

    assert len(set(raw_data1.columns.values) - set(raw_data2.columns.values)) == 0

    Y1 = ['N' for _ in range(0, raw_data1.shape[0])]
    Y2 = ['Y' for _ in range(0, raw_data2.shape[0])]

    print raw_data1.shape
    print raw_data2.shape

    #print Y1 + Y2
    Y = Y1 + Y2
    X = pd.concat([raw_data1, raw_data2])
    print X.shape

    #print X.as_matrix()
    #print raw_data1.columns[139]

    #clf = tree.DecisionTreeClassifier()
    #clf = clf.fit(X, Y)
    #dot_data = tree.export_graphviz(clf, out_file=None)
    #graph = pydotplus.graph_from_dot_data(dot_data)
    #graph.write_pdf("tree.pdf")

    #X_new = SelectKBest(chi2, k=10).fit_transform(X, Y)
    #print X_new.shape

    X['Y'] = Y
    print X.shape
    X.to_csv('classifier.csv', index=False)
