import pandas as pd

def merge_bank_rank_events(df):
    read = ["UNC_M_RD_CAS_RANK{}.BANK{}".format(i,j) for i in range(0,8) for j in range(0,8) ]
    counts = []
    for re in read:
        counts.append(df[re])
    df['ENG.UNC_M_RD_CAS.SUM'] = np.sum(counts)
    df['ENG.UNC_M_RD_CAS.STD'] = np.std(counts, ddof=0)

    write = ["UNC_M_WR_CAS_RANK{}.BANK{}".format(i,j) for i in range(0,8) for j in range(0,8) ]
    counts = []
    for we in write:
        counts.append(df[we])
    df['ENG.UNC_M_WR_CAS.SUM'] = np.sum(counts)
    df['ENG.UNC_M_WR_CAS.STD'] = np.std(counts, ddof=0)

    print df['ENG.UNC_M_RD_CAS.SUM']
    print df['ENG.UNC_M_RD_CAS.STD']
    print df['ENG.UNC_M_WR_CAS.SUM']
    print df['ENG.UNC_M_WR_CAS.STD']

def aggregation_matrix(prefix, series):
    matrix = pd.DataFrame(series)
    matrix.reset_index(inplace=True)
    pivot_table = matrix.pivot(index='INDEX', columns='EVENT_NAME', values='SAMPLE_VALUE')
    pivot_table.rename(columns=lambda x: "{}.{}".format(prefix, x), inplace=True)
    return pivot_table

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

    # Aggregate all event samples from the same event at time
    aggregates = []
    if aggregate_samples:
        grouped_df = raw_data.groupby(['EVENT_NAME', 'INDEX'])
        for agg in aggregate_samples:
            if agg == 'mean':
                series = grouped_df['SAMPLE_VALUE'].mean()
                aggregates.append(aggregation_matrix('AVG', series))
            elif agg == 'std':
                series = grouped_df['SAMPLE_VALUE'].std(ddof=0)
                aggregates.append(aggregation_matrix('STD', series))
            elif agg == 'max':
                series = grouped_df['SAMPLE_VALUE'].max()
                aggregates.append(aggregation_matrix('MAX', series))
            elif agg == 'min':
                series = grouped_df['SAMPLE_VALUE'].min()
                aggregates.append(aggregation_matrix('MIN', series))
            else:
                assert "Unknown aggregation: {}. Supported are: [mean, std, max, min].".format(agg)

    df = pd.concat(aggregates, axis=1)

    # Remove events whose deltas are all 0:
    if remove_zero:
        df = df.drop(get_all_zero_events(df))

    # Cut off everything after first row with a NaN value
    if cut_off_nan:
        min_idx = minimum_nan_index(df)
        throw_away = df.shape[0]-min_idx
        if throw_away > df.shape[0] * (0.20):
            print("Throwing away {} out of {} samples for {}".format(throw_away, df.shape[0], f))
        df = df[:min_idx]

    return df[1:] # Throw away the first sample because we don't know at what time the ROI started...

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
    nans = pd.isnull(df).any(1).nonzero()[0]
    if len(nans) == 0:
        return df.shape[0]
    else:
        return min(nans)

def get_zero_features_in_matrix(df):
    """
    Given a pandas DataFrame loaded from a matrix_X*.csv file,
    return all columns (features) where the values are always zero.
    """
    zero_events = []
    for col in df:
        if not df[col].any():
            # col.split(".", 1)[1] for getting event name
            zero_events.append(col)
    return zero_events

def get_all_zero_events(df):
    """
    Given a pandas DataFrame loaded from a results.csv file,
    return all event names where the counts are always 0
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
