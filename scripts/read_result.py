#!/usr/bin/env python

import sys
import pandas as pd


filename = sys.argv[1]

df = pd.read_csv(filename, index_col=0, skipinitialspace=True)

#print df.loc['UOPS_EXECUTED.CYCLES_GE_1_UOP_EXEC']

for idx in df.index:
    print idx, df.loc[[idx], 'SAMPLE_VALUE'].tolist()

#print df.get_values()
