import math
import pandas as pd
import numpy as np

READ_BANK_EVENTS = ["UNC_M_RD_CAS_RANK{}.BANK{}".format(i,j) for i in range(0,8) for j in range(0,8) ]
WRITE_BANK_EVENTS = ["UNC_M_WR_CAS_RANK{}.BANK{}".format(i,j) for i in range(0,8) for j in range(0,8) ]

def merge_bank_rank_events(df, minmax=False):
    matrix = pd.DataFrame(df)
    matrix.reset_index(inplace=True)
    pivot_table = matrix.pivot(index='INDEX', columns='EVENT_NAME', values='SAMPLE_VALUE')
    df = pivot_table

    read_rank_banks = pd.DataFrame()
    for e in READ_BANK_EVENTS:
        read_rank_banks[e] = df[e]

    write_rank_banks = pd.DataFrame()
    for e in WRITE_BANK_EVENTS:
        write_rank_banks[e] = df[e]

    merged_banks = pd.DataFrame()
    merged_banks['SUM.UNC_M_RD_CAS.*'] = read_rank_banks.sum(axis=1)
    merged_banks['STD.UNC_M_RD_CAS.*'] = read_rank_banks.std(axis=1, ddof=0)
    merged_banks['SUM.UNC_M_WR_CAS.*'] = write_rank_banks.sum(axis=1)
    merged_banks['STD.UNC_M_WR_CAS.*'] = write_rank_banks.std(axis=1, ddof=0)
    if minmax:
        merged_banks['MAX.UNC_M_WR_CAS.*'] = write_rank_banks.max(axis=1)
        merged_banks['MIN.UNC_M_WR_CAS.*'] = write_rank_banks.min(axis=1)
        merged_banks['MAX.UNC_M_RD_CAS.*'] = read_rank_banks.max(axis=1)
        merged_banks['MIN.UNC_M_RD_CAS.*'] = read_rank_banks.min(axis=1)
    #print(merged_banks)
    return merged_banks

def add_metrics(df):
    matrix = pd.DataFrame(df)
    matrix.reset_index(inplace=True)
    pivot_table = matrix.pivot(index='INDEX', columns='EVENT_NAME', values='SAMPLE_VALUE')
    df = pivot_table

    metrics = pd.DataFrame()
    metrics['ENG.IPC'] = pivot_table['INST_RETIRED.ANY_P'] / pivot_table['CPU_CLK_UNHALTED.THREAD_P_ANY']
    metrics['ENG.DSB_SWITCHES'] = pivot_table['DSB2MITE_SWITCHES.PENALTY_CYCLES'] / pivot_table['CPU_CLK_UNHALTED.THREAD_P_ANY']
    metrics['ENG.MS_SWITCHES'] = 3 * pivot_table['IDQ.MS_SWITCHES'] / pivot_table['CPU_CLK_UNHALTED.THREAD_P_ANY']
    metrics['ENG.L2_BOUND'] = (pivot_table['CYCLE_ACTIVITY.STALLS_L1D_PENDING'] - pivot_table['CYCLE_ACTIVITY.STALLS_L2_PENDING']) / pivot_table['CPU_CLK_UNHALTED.THREAD_P_ANY']
    metrics['ENG.L3_HIT_FRACTION'] = pivot_table['MEM_LOAD_UOPS_RETIRED.LLC_HIT'] / (pivot_table['MEM_LOAD_UOPS_RETIRED.LLC_HIT']+7*pivot_table['MEM_LOAD_UOPS_RETIRED.LLC_MISS'])
    metrics['ENG.L3_BOUND'] = (metrics['ENG.L3_HIT_FRACTION'] * pivot_table['CYCLE_ACTIVITY.STALLS_L2_PENDING']) / pivot_table['CPU_CLK_UNHALTED.THREAD_P_ANY']
    metrics['ENG.MEM_BOUND'] = ((1 - metrics['ENG.L3_HIT_FRACTION']) * pivot_table['CYCLE_ACTIVITY.STALLS_L2_PENDING']) / pivot_table['CPU_CLK_UNHALTED.THREAD_P_ANY']
    #metrics['ENG.STALLS_MEM_ANY'] = pd.concat(pivot_table['CPU_CLK_UNHALTED.THREAD'], pivot_table['CYCLE_ACTIVITY.STALLS_L1D_PENDING']).min(axis=1)
    #metrics['ENG.STORES_BOUND'] = (pivot_table['RESOURCE_STALLS.SB'] - metrics['ENG.STALLS_MEM_ANY']) / pivot_table['CPU_CLK_UNHALTED.THREAD_P_ANY']

    return metrics

def aggregation_matrix(prefix, series, drop_bank_events=False):
    matrix = pd.DataFrame(series)
    matrix.reset_index(inplace=True)
    pivot_table = matrix.pivot(index='INDEX', columns='EVENT_NAME', values='SAMPLE_VALUE')
    if drop_bank_events:
        pivot_table.drop(READ_BANK_EVENTS, axis=1, inplace=True)
        pivot_table.drop(WRITE_BANK_EVENTS, axis=1, inplace=True)

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
    raw_data.sort_index(inplace=True)

    # Convert time
    time_to_ms(raw_data)

    # Aggregate all event samples from the same event at time
    aggregates = []
    drop_bank_events = 'rbmerge' in aggregate_samples or 'rbmerge2' in aggregate_samples or 'rbdrop' in aggregate_samples

    start_at = 0
    if aggregate_samples:
        grouped_df = raw_data.groupby(['EVENT_NAME', 'INDEX'])
        grouped_df_multiple = grouped_df.filter(lambda x: len(x) > 1).groupby(['EVENT_NAME', 'INDEX'])
        for agg in aggregate_samples:
            if agg == 'mean':
                series = grouped_df['SAMPLE_VALUE'].mean()
                aggregates.append(aggregation_matrix('AVG', series, drop_bank_events=drop_bank_events))
            elif agg == 'std':
                series = grouped_df_multiple['SAMPLE_VALUE'].std(ddof=0)
                matrix = aggregation_matrix('STD', series, drop_bank_events=drop_bank_events)
                aggregates.append(matrix)
            elif agg == 'max':
                series = grouped_df_multiple['SAMPLE_VALUE'].max()
                aggregates.append(aggregation_matrix('MAX', series, drop_bank_events=drop_bank_events))
            elif agg == 'min':
                series = grouped_df_multiple['SAMPLE_VALUE'].min()
                aggregates.append(aggregation_matrix('MIN', series, drop_bank_events=drop_bank_events))
            elif agg == 'rbmerge':
                series = grouped_df['SAMPLE_VALUE'].mean()
                aggregates.append(merge_bank_rank_events(series))
            elif agg == 'rbmerge2':
                series = grouped_df['SAMPLE_VALUE'].mean()
                aggregates.append(merge_bank_rank_events(series, minmax=True))
            elif agg == 'cut1':
                start_at = 1
            elif agg == 'cut2':
                start_at = 2
            elif agg == 'cut4':
                start_at = 4
            elif agg == 'rbdrop':
                pass
            elif agg == 'metrics':
                series = grouped_df['SAMPLE_VALUE'].mean()
                aggregates.append(add_metrics(series))
            else:
                assert "Unknown aggregation: {}. Supported are: [mean, std, max, min, rbmerge, cut1, cut2, cut4].".format(agg)
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

    if "merge4" in aggregate_samples:
        # Aggregate 4 rows to get 1sec sampling time
        df.reset_index(inplace=True)
        df['MergeLabel'] = pd.Series([ math.ceil(i / 4.0) for i in range(1, len(df)+1) ])
        df = df.groupby(['MergeLabel']).sum()

    if "merge2" in aggregate_samples:
        # Aggregate 2 rows to get 0.5sec sampling time
        df.reset_index(inplace=True)
        df['MergeLabel'] = pd.Series([ math.ceil(i / 2.0) for i in range(1, len(df)+1) ])
        df = df.groupby(['MergeLabel']).sum()

    return df[start_at:]


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
