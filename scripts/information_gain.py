def information_gain(x, y):

    def _entropy(values):
        counts = np.bincount(values)
        probs = counts[np.nonzero(counts)] / float(len(values))
        return - np.sum(probs * np.log(probs))

    def _information_gain(feature, y):
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
