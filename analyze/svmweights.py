import os
import sys

import pandas as pd
import numpy as np
import sklearn

from jinja2 import Template, Environment, FileSystemLoader
from flask import request

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from classify import get_argument_parser, svm

parser = get_argument_parser("autoperf Viewer arguments")
args = parser.parse_args()

try:
  config = curdoc().session_context.request.arguments['config'][0]
  app = curdoc().session_context.request.arguments['app'][0]
except:
  config = 'L3-SMT-cores'
  app = 'AA700'

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

X, Y, _, X_test, Y_test = svm.row_training_and_test_set(args.data_directory, args.config, [app], cutoff=args.cutoff, uncore=args.uncore)
clf = svm.CLASSIFIERS['poly2balanced']

min_max_scaler = sklearn.preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
X_test_scaled = min_max_scaler.transform(X_test)
clf.fit(X_scaled, Y)
Y_pred = clf.predict(X_test_scaled)
metrics = svm.get_svm_metrics(args, [app], Y, Y_test, Y_pred)
source = ColumnDataSource(ColumnDataSource.from_df(pd.DataFrame([metrics])))

columns = [
        TableColumn(field="Samples Training 1", title="Samples Training 1"),
        TableColumn(field="Samples Training 0", title="Samples Training 0"),
        TableColumn(field="Samples Test 0", title="Samples Test 0"),
        TableColumn(field="Recall", title="Recall"),
        TableColumn(field="Error", title="Error"),
        TableColumn(field="Precision", title="Precision"),
        TableColumn(field="Tested Application", title="Tested Application"),
        TableColumn(field="Samples Test Total", title="Samples Test Total"),
        TableColumn(field="F1 score", title="F1"),
        TableColumn(field="Training Configs", title="Training Configs"),
        TableColumn(field="Samples Test 1", title="Samples Test 1"),
        TableColumn(field="Samples Training Total", title="Samples Training Total"),
        TableColumn(field="Accuracy", title="Accuracy"),
]

data_table = DataTable(source=source, width=1024)

env = Environment(loader=FileSystemLoader(os.path.join(sys.path[0], "templates")))
env.globals['folders'] = get_menu()
template = env.get_template('svmweights.html')

widgets = row(data_table, sizing_mode='scale_width')

curdoc().add_root(widgets)
curdoc().title = "Distribution for high-weight SVM features"
curdoc().template = template
