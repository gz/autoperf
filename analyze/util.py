import pandas as pd

def load_as_X(f, aggregate_samples=['mean'], remove_zero=False, cut_off_nan=True):
    """
    Transform CSV file into a matrix X (used for most ML inputs).
    The rows will be different times, the columns are the events.

    Keyword arguments:
    aggregate_samples -- Aggregate samples from all CPUs at time t.
    remove_zero -- Remove features that are all zero.
    cut_off_nan -- Remove everything after first NaN value is encountered.
    """
    # Parse file
    raw_data = pd.read_csv(f, sep=',', skipinitialspace=True)
    raw_data.sortlevel(inplace=True)

    # Convert time
    time_to_ms(raw_data)
    #print raw_data

    # Cut off everything after first NaN value:
    if cut_off_nan:
        df = raw_data.groupby(['EVENT_NAME', 'TIME']).count()
        df.reset_index(level=['TIME'], inplace=True)
        sample_lengths = [len(df.loc[group, :]) for group in df.index.unique()]
        cutoff = min(sample_lengths)
        max_samples = max(sample_lengths)
        if cutoff + 30 < max_samples:
            print("Limiting to {} max is {}".format(cutoff, max_samples))


    # Aggregate all event samples from the same event at time
    if aggregate_samples:
        aggregates = []
        grouped_df = raw_data.groupby(['EVENT_NAME', 'TIME'])
        if 'mean' in aggregate_samples:
            df_mean = grouped_df.mean()
            df_mean.rename(lambda event: "AVG.{}".format(event), inplace=True)
            aggregates.append(df_mean)
        if 'std' in aggregate_samples:
            df_std = grouped_df.std(ddof=0)
            df_std.rename(lambda event: "STD.{}".format(event), inplace=True)
            aggregates.append(df_std)
        if 'max' in aggregate_samples:
            df_max = grouped_df.max()
            df_max.rename(lambda event: "MAX.{}".format(event), inplace=True)
            aggregates.append(df_max)
        if 'min' in aggregate_samples:
            df_min = grouped_df.min()
            df_min.rename(lambda event: "MIN.{}".format(event), inplace=True)
            aggregates.append(df_min)
        if len(aggregates) == 0:
            assert "Unknown aggregation: {}. Supported are: [mean, std, max, min].".format(aggregate_samples)

        df = pd.concat(aggregates, axis=0)

    df.reset_index(level=['TIME'], inplace=True)

    # Remove events whose deltas are all 0:
    if remove_zero:
        df = df.drop(get_all_zero_events(df))

    df = result_to_matrix(df, cutoff)
    #for idx, has_null in df.isnull().any(axis=1).items():
    #    if has_null:
    #        print("FOUND NaN!")
    #        assert "found nan in ", idx

    return df

def result_to_matrix(df, cutoff):
    """
    Transform the result as read from results.csv in a matrix of the following format:

    EVENT1  EVENT2  EVENT3 .... EVENTN
    12           9       5          12
     1           1       2           5
     0         NaN     100          12
     0         NaN     NaN          99

    Note: NaN numbers may appear for an event at the end in case the individual events
    can be read from different runs containing a different amount of samples.
    Differences of just a few samples is normally not a problem. Big discrepancies
    would indicate unstable runtimes of your algorithm.
    """
    frames = []
    print "result to matrix"
    for idx in df.index.unique():
        series = df.loc[[idx], 'SAMPLE_VALUE'].head(cutoff)
        new_series = series.rename(idx).reset_index(drop=True)
        frames.append(new_series)

    # Column i is event i
    print "concat"
    matrix = pd.concat(frames, axis=1)
    print "result to matrix done"
    return matrix

def minimum_nan_index(df):
    """
    Return the earliest index that contains NaN over all columns or None
    if there are no NaN values in any columns.

    # Example
    For the following matrix it returns 1 as (1,1) is NaN:
    idx | EVENT1   EVENT2  EVENT3 .... EVENTN
      0 |     12        9       5          12
      1 |      1      NaN       2           5
      2 |      0      NaN     100          12
      3 |      0      NaN       1          99
    """
    for idx, has_null in df.isnull().any(axis=1).items():
        if has_null:
            print("found nan in ", idx)
    for idx, has_null in df.isnull().any(axis=1).items():
        if has_null:
            print("nan offset", idx)
            return idx


def get_all_zero_events(df):
    """
    Given a pandas DataFrame loaded from a results.csv file,
    return all event names where the samples sum up to 0
    """
    event_names = []
    for idx in df.index.unique():
        if df.loc[idx, 'SAMPLE_VALUE'].sum() == 0:
            event_names.append(idx)
    return event_names

def time_to_ms(df):
    """
    Transforn the perf time (floating point, seconds)
    to miliseconds (absolute numbers)
    """
    df['TIME'] = df['TIME'].map(lambda x: int(x * 1000))
