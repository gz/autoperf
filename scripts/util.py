import pandas as pd

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
