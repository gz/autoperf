import sys
import re
import pandas as pd

def get_selected_events(weka_fold_file):
    df = pd.DataFrame()
    with open(weka_fold_file) as f:
        for line in f.readlines():
            regex = r"\s+(\d+)\(\s*(\d+)\s+%\)\s+(\d+)\s+(.*)"
            matches = re.match(regex, line)
            if matches:
                fold = int(matches.group(1))
                index = matches.group(3)
                event = matches.group(4)

                row = { 'column_index': index, 'name': event, 'folds': fold }
                df = df.append(row, ignore_index=True)
    return df


if __name__ == '__main__':
    df = get_selected_events(sys.argv[1])
    print "EVENTS"
    print "============================="
    df_filtered = df[df.folds >= 5]
    print df_filtered
    print len(df_filtered)
