import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot


def ig(X, y):

    def get_t1(fc, c, f):
        t = np.log2(fc/(c * f))
        t[~np.isfinite(t)] = 0
        return np.multiply(fc, t)

    def get_t2(fc, c, f):
        t = np.log2((1-f-c+fc)/((1-c)*(1-f)))
        t[~np.isfinite(t)] = 0
        return np.multiply((1-f-c+fc), t)

    def get_t3(c, f, class_count, observed, total):
        nfc = (class_count - observed)/total
        t = np.log2(nfc/(c*(1-f)))
        t[~np.isfinite(t)] = 0
        return np.multiply(nfc, t)

    def get_t4(c, f, feature_count, observed, total):
        fnc = (feature_count - observed)/total
        t = np.log2(fnc/((1-c)*f))
        t[~np.isfinite(t)] = 0
        return np.multiply(fnc, t)

    X = check_array(X, accept_sparse='csr')
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    # counts

    observed = safe_sparse_dot(Y.T, X)          # n_classes * n_features
    total = observed.sum(axis=0).reshape(1, -1).sum()
    feature_count = X.sum(axis=0).reshape(1, -1)
    class_count = (X.sum(axis=1).reshape(1, -1) * Y).T

    # probs

    f = feature_count / feature_count.sum()
    c = class_count / float(class_count.sum())
    fc = observed / total

    # the feature score is averaged over classes
    scores = (get_t1(fc, c, f) +
            get_t2(fc, c, f) +
            get_t3(c, f, class_count, observed, total) +
            get_t4(c, f, feature_count, observed, total)).mean(axis=0)

    scores = np.asarray(scores).reshape(-1)

    return scores, []
