import os

import pandas as pd
from flask import request

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.plotting import figure
from bokeh.sampledata.unemployment1948 import data


import util

try:
  config = curdoc().session_context.request.arguments['config'][0]
  folder = curdoc().session_context.request.arguments['folder'][0]
except:
  config = 'L3-SMT'
  folder = 'AA700'

RESULTS_BASE = '/home/gz/workspace/results-babybel/'
results_file = os.path.join(RESULTS_BASE, config, folder, "results_uncore_shared.csv")
df_matrix = pd.read_csv(os.path.join(RESULTS_BASE, config, folder, "matrix_X_uncore_shared.csv"))

df_matrix = util.load_as_X(results_file, aggregate_samples='mean', remove_zero=True, cut_off_nan=True)
#print df_matrix.columns
#print df_matrix.index

correlation_matrix = df_matrix.corr()
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors)

cols = []
rows = []
all_correlation = []
for (row_index, correlation) in correlation_matrix.iterrows():
    for col_num, corr in enumerate(correlation):
        rows.append(row_index)
        cols.append(correlation_matrix.columns[col_num])
        all_correlation.append(corr)

#print "cols", len(correlation_matrix.columns.values)
#print "rows", len(correlation_matrix.index.values)
#print "all", len(correlation_matrix.columns.values) * len(correlation_matrix.index.values)
#print "correlation len", len(correlation)

source = ColumnDataSource(
    data=dict(columns=cols, rows=rows, correlation=all_correlation)
)

TOOLS = "hover,save,pan,box_zoom,wheel_zoom"

p = figure(title="Event correlation",
           x_range=correlation_matrix.columns.values.tolist(), y_range=correlation_matrix.index.values.tolist(),
           x_axis_location="above", plot_width=1024, plot_height=1024,
           tools=TOOLS, webgl=True)

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "1pt"
p.axis.major_label_standoff = 0
from math import pi
p.xaxis.major_label_orientation = pi / 3

p.rect(x="columns", y="rows", width=1, height=1,
       source=source,
       fill_color={'field': 'correlation', 'transform': mapper},
       line_color=None)

p.select_one(HoverTool).tooltips = [
    ('Compare', '@rows vs. @columns'),
    ('correlation', '@correlation'),
]

curdoc().add_root(p)
curdoc().title = "Clustering"
