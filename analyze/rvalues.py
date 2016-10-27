import os
import sys

import pandas as pd
import numpy as np
import sklearn

from jinja2 import Template, Environment, FileSystemLoader
from flask import request

from bokeh.io import curdoc, vplot
from bokeh.layouts import column
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn, Select
from bokeh.models import ColumnDataSource, Range1d
from bokeh.plotting import figure
from bokeh.charts import Histogram

sys.path.insert(1, os.path.join(os.path.realpath(os.path.split(__file__)[0]), '..'))
from analyze.classify import get_argument_parser, svm

parser = get_argument_parser("autoperf Viewer arguments")
args = parser.parse_args()

try:
  config = str(curdoc().session_context.request.arguments['config'][0], 'utf-8')
  app = str(curdoc().session_context.request.arguments['app'][0], 'utf-8')
except:
  config = 'L3-SMT'
  app = 'AA700'

X, Y, _, X_test, Y_test = svm.row_training_and_test_set(args.data_directory, [config], [app], cutoff=args.cutoff, uncore=args.uncore)

def get_menu():
    dirs = {'L3-SMT': [], 'L3-SMT-cores': []}
    for directory, subdir, files in os.walk(args.data_directory):
        if os.path.exists(os.path.join(directory, 'completed')):
            path, pair = os.path.split(directory)
            _, config = os.path.split(path)
            dirs[config].append(pair.split("_vs_")[0])

    dirs['L3-SMT'] = sorted(set(dirs['L3-SMT']))
    dirs['L3-SMT-cores'] = sorted(set(dirs['L3-SMT-cores']))
    return dirs

def get_r_values(df):

    yes = df[Y==True]
    no  = df[Y==False]
    yes = yes.reset_index()
    no = no.reset_index()

    Rs = []
    for col in df:
        if df[col].sum() == 0:
            continue

        yes_event = yes[col]
        no_event = no[col]

        miny = yes_event.min()
        maxy = yes_event.max()
        minn = no_event.min()
        maxn = no_event.max()

        min_both = max(miny, minn)
        max_both = min(maxy, maxn)
        if max_both == min_both:
            Rs.append( { 'Event': col, 'R': 1, 'Ry': 1, 'Rn': 1 } )
            continue
        elif max_both < min_both:
            Rs.append( { 'Event': col, 'R': 0, 'Ry': 0, 'Rn': 0 } )
            continue

        county = yes_event[yes_event >= min_both][yes_event <= max_both].count()
        countn = no_event[yes_event >= min_both][yes_event <= max_both].count()

        total_yes_no = yes_event.count() + no_event.count()
        R = (county+countn) / total_yes_no
        Ry = county / yes_event.count()
        Rn = countn / no_event.count()
        Rs.append( { 'Event': col, 'R': R, 'Ry': Ry, 'Rn': Rn } )

    return Rs

env = Environment(loader=FileSystemLoader(os.path.join(sys.path[0], "templates")))
env.globals['folders'] = get_menu()
env.globals['config'] = config
env.globals['app'] = app
template = env.get_template('rvalues.html')

columns = [
     TableColumn(field="Event", title="Event", width=500),
     TableColumn(field="R", title="R", width=180),
     TableColumn(field="Ry", title="Ry", width=180),
     TableColumn(field="Rn", title="Rn", width=180),
]

df = pd.DataFrame( get_r_values(X) )
df.to_csv("r_values_nbody.csv")

source = ColumnDataSource(ColumnDataSource.from_df(pd.DataFrame( get_r_values(X) )))
data_table = DataTable(source=source, columns=columns, fit_columns=True, width=1400, height=800)

curdoc().add_root(data_table)
curdoc().title = "R values for SVM features"
curdoc().template = template
