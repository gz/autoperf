import os
import sys

import pandas as pd
import numpy as np

from jinja2 import Template, Environment, FileSystemLoader
from bokeh.io import curdoc
from bokeh.layouts import gridplot, column, row
from bokeh.charts import Line, show, output_file
from bokeh.models.widgets import PreText, Select
from bokeh.models import ColumnDataSource, Range1d
from bokeh.plotting import figure
from bokeh.palettes import Spectral11, Blues4

from flask import request

if curdoc().session_context.request.arguments.has_key('config'):
    config = curdoc().session_context.request.arguments['config'][0]
if curdoc().session_context.request.arguments.has_key('folder'):
    folder = curdoc().session_context.request.arguments['folder'][0]

RESULTS_BASE = '/home/gz/workspace/results-babybel/'

df_csv = pd.read_csv(os.path.join(RESULTS_BASE, config, folder, "results_uncore_shared.csv"))

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
    events = df['EVENT_NAME'].values
    return sorted(list(set(events)))

def get_xs_ys_for(event_name):
    xs = []
    ys = []
    relevant_events =  df_csv[df_csv['EVENT_NAME'] == event_name]
    relevant_events_by_cpu = relevant_events.set_index('CPU')
    for idx in relevant_events_by_cpu.index.unique():
        xs.append(relevant_events_by_cpu.loc[idx]['TIME'])
        ys.append(relevant_events_by_cpu.loc[idx]['SAMPLE_VALUE'])
    return xs, ys

def get_series_for(event_name):
    events = df_csv[df_csv.EVENT_NAME == event_name]
    minimum = events.groupby('TIME').min()['SAMPLE_VALUE']
    mean = events.groupby('TIME').mean()['SAMPLE_VALUE']
    maximum = events.groupby('TIME').max()['SAMPLE_VALUE']
    ts = mean.index
    left = mean.index - 0.125
    right = mean.index + 0.125
    return ts, mean, maximum, minimum, left, right

all_events = get_events(df_csv)

ticker = Select(value=all_events[0], options=all_events, width=500)

x, average, maximum, minimum, left, right = get_series_for(all_events[0])
avg_source = ColumnDataSource( data=dict(x=x, average=average, max=maximum, min=minimum, left=left, right=right) )

xs, ys = get_xs_ys_for(all_events[0])
cpus_source = ColumnDataSource(dict(xs=xs, ys=ys, color=Spectral11))

def ticker_change(attrname, old, new):
    x, average, maximum, minimum, left, right = get_series_for(new)
    avg_source.data = dict(x=x, average=average, max=maximum, min=minimum, left=left, right=right)
    xs, ys = get_xs_ys_for(new)
    cpus_source.data = dict(xs=xs, ys=ys, color=Spectral11)

ticker.on_change('value', ticker_change)

fig_avg = figure()
variation = fig_avg.quad(top='max', bottom='min', left='left', right='right', source=avg_source, color=Blues4[1], legend="Max/Min", alpha=0.5)
line = fig_avg.line(x='x', y='average', source=avg_source, line_width=5, legend="Average", color=Blues4[0])

fig_cpus = figure(title='Individual CPUs')
cpus_plot = fig_cpus.multi_line(xs='xs', ys='ys', source=cpus_source, line_width=3, line_color='color')

widgets = column(ticker, column(fig_avg, fig_cpus, sizing_mode='scale_width'), sizing_mode='scale_width')
env = Environment(loader=FileSystemLoader(os.path.join(sys.path[0], "templates")))
env.globals['folders'] = get_menu()
template = env.get_template('events.html')

curdoc().add_root(widgets)
curdoc().title = "Events"
curdoc().template = template
