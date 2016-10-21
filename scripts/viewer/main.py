import os
import sys

import pandas as pd
import numpy as np

from jinja2 import Template, Environment, FileSystemLoader
from bokeh.io import curdoc
from bokeh.layouts import gridplot, column, row
from bokeh.charts import Line, show, output_file
from bokeh.models.widgets import PreText, Select
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from flask import request

config = curdoc().session_context.request.arguments['config'][0]
folder = curdoc().session_context.request.arguments['folder'][0]

RESULTS_BASE = '/home/zgerd/workspace/results-babybel/'

df_csv = pd.read_csv(os.path.join(RESULTS_BASE, config, folder, "results_uncore_shared.csv"))
df = pd.read_csv(os.path.join(RESULTS_BASE, config, folder, "matrix_X_uncore_shared.csv"))

def get_matrix_file(results_path):
    MATRIX_FILE = 'matrix_X_uncore_{}.csv'.format(args.uncore)

    matrix_file = os.path.join(results_path, MATRIX_FILE)
    if os.path.exists(os.path.join(results_path, 'completed')):
        if os.path.exists(matrix_file):
            return matrix_file
        else:
            print "No matrix file ({}) found, run the scripts/pair/matrix_all.py script first!".format(matrix_file)
            sys.exit(1)
    else:
        print "Unfinished directory: {}".format(results_path)
        sys.exit(1)

def get_menu():
    dirs = {'L3-SMT': [], 'L3-SMT-cores': []}
    for directory, subdir, files in os.walk(RESULTS_BASE):
        if os.path.exists(os.path.join(directory, 'completed')):
            path, pair = os.path.split(directory)
            _, config = os.path.split(path)
            dirs[config].append(pair)

    dirs['L3-SMT'] = sorted(dirs['L3-SMT'])
    dirs['L3-SMT-cores'] = sorted(dirs['L3-SMT-cores'])
    return dirs

def get_events(df):
    print df.columns.values
    events = [ val.split('.', 1)[1] for val in df.columns.values ]
    return sorted(list(set(events)))

def avg_index(event):
    return "AVG.{}".format(event)

def std_index(event):
    return "STD.{}".format(event)

all_events = get_events(df)


ticker = Select(value=all_events[0], options=all_events, width=500)

source_avg = ColumnDataSource(data=dict(x=df.index.values, y=df[avg_index(all_events[0])]))
source_std = ColumnDataSource(data=dict(x=df.index.values, y=df[std_index(all_events[0])]))

def ticker_change(attrname, old, new):
    source_avg.data = dict(x=df.index.values, y=df[avg_index(new)])
    source_std.data = dict(x=df.index.values, y=df[std_index(new)])

ticker.on_change('value', ticker_change)

fig_avg = figure(title='Average', height=300)
line = fig_avg.line('x', 'y', source=source_avg, line_width=5)

fig_std = figure(title='Standard Deviation', height=300)
line = fig_std.line('x', 'y', source=source_std, line_width=5)


xs = []
ys = []
def get_xs_ys_for(event_name):
    print df_csv
    relevant_events =  df_csv[df_csv['EVENT_NAME'] == 'UOPS_EXECUTED.CORE_CYCLES_GE_2']

    relevant_events.set_index('CPU')
    for idx in relevant_events.index.values:
        print idx
        xs.append(relevant_events.loc[idx]['TIME'])
        ys.append(relevant_events.loc[idx]['SAMPLE_VALUE'])
    return xs, ys

xs, ys = get_xs_ys_for('UOPS_EXECUTED.CORE_CYCLES_GE_2')
cpus_source = ColumnDataSource(dict(xs=xs, ys=ys))
fig_cpus = figure(title='Individual CPUs', height=300)
cpus_plot = fig_cpus.multi_line(xs='xs', ys='ys', source=cpus_source, line_width=5)

widgets = column(ticker, column(
                            row(fig_avg, fig_std, sizing_mode='scale_width'),
                            fig_cpus, sizing_mode='scale_width'), sizing_mode='scale_width')


env = Environment(loader=FileSystemLoader(os.path.join(sys.path[0], "templates")))
env.globals['folders'] = get_menu()
template = env.get_template('events.html')

curdoc().add_root(widgets)
curdoc().title = "Events"
curdoc().template = template
