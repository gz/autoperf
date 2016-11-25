#!/usr/bin/env python3

import os
import sys

import pandas as pd

if __name__ == '__main__':
    print("Finding min error heatmap")
    heatmaps = []
    for root, dirs, files in os.walk(sys.argv[1]):
        for name in files:
            p = os.path.join(root, name)
            if name.endswith(".csv") and name.startswith("svm_heatmap_"):
                df = pd.read_csv(p)
                heatmaps.append( (len(df[df.Accuracy > 0.70]), p) )

    for score, path in sorted(heatmaps):
        print (score, path)
