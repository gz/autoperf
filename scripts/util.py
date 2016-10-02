import pandas as pd

def load_as_X(f, aggregate_samples=True, remove_zero=True, cut_off_nan=True):
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

    # Aggregate all event samples from the same event at time
    if aggregate_samples:
        grouped_df = raw_data.groupby(['EVENT_NAME', 'TIME']).sum()
        grouped_df.reset_index(level=['TIME'], inplace=True)

    # Remove events whose deltas are all 0:
    if remove_zero:
        df = grouped_df.drop(get_all_zero_events(grouped_df))
        df = result_to_matrix(df)

    # Cut off everything after first NaN value:
    if cut_off_nan:
        cut_off = minimum_nan_index(df)
        if cut_off != None:
            df = df[:cut_off]

    return df

def result_to_matrix(df):
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
    for idx in df.index.unique():
        series = df.loc[[idx], 'SAMPLE_VALUE']
        new_series = series.rename(idx).reset_index(drop=True)
        frames.append(new_series)

    # Column i is event i
    return pd.concat(frames, axis=1)

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
    for idx, has_null in df.isnull().any(axis=1).iteritems():
        if has_null:
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
