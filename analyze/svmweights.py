import os
import sys

import pandas as pd
import numpy as np
import sklearn

from jinja2 import Template, Environment, FileSystemLoader
from flask import request

from bokeh.io import curdoc, vplot
from bokeh.layouts import gridplot
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn, Select
from bokeh.models import ColumnDataSource
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

X, Y, _, X_test, Y_test = svm.row_training_and_test_set(args.data_directory, [config], [app], cutoff=args.cutoff, uncore=args.uncore)
clf = svm.CLASSIFIERS['linear']
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
X_test_scaled = min_max_scaler.transform(X_test)
model = clf.fit(X_scaled, Y)
Y_pred = clf.predict(X_test_scaled)
metrics = svm.get_svm_metrics(args, [app], Y, Y_test, Y_pred)
## source = ColumnDataSource(ColumnDataSource.from_df(pd.DataFrame([metrics])))

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

abs_coefs = np.abs(model.coef_[0])
weight_idx_sorted = np.argsort( abs_coefs )
weights = list(weight_idx_sorted)
weights.reverse()

##weights_to_select = [ "{} ({:.2f})".format(X_test.columns[idx], abs_coefs[idx]) for idx in weights ]
##ticker = Select(value=weights_to_select[0], options=weights_to_select, width=500)
## columns = [
##         TableColumn(field="Samples Training 1", title="Samples Training 1"),
##         TableColumn(field="Samples Training 0", title="Samples Training 0"),
##         TableColumn(field="Samples Test 0", title="Samples Test 0"),
##         TableColumn(field="Recall", title="Recall"),
##         TableColumn(field="Error", title="Error"),
##         TableColumn(field="Precision", title="Precision"),
##         TableColumn(field="Tested Application", title="Tested Application"),
##         TableColumn(field="Samples Test Total", title="Samples Test Total"),
##         TableColumn(field="F1 score", title="F1"),
##         TableColumn(field="Training Configs", title="Training Configs"),
##         TableColumn(field="Samples Test 1", title="Samples Test 1"),
##         TableColumn(field="Samples Training Total", title="Samples Training Total"),
##         TableColumn(field="Accuracy", title="Accuracy"),
## ]
##
## data_table = DataTable(source=source, columns=columns, height=)

def get_histogram_data(index):
    top_event = X_scaled_df.columns[index]

    Y_labels = Y.map(lambda x: "Training Y" if x else "Training N")
    Y_labels = Y_labels.reset_index()
    distribution = pd.DataFrame()
    distribution['value'] = X_scaled_df[top_event]
    distribution = distribution.reset_index()
    del distribution['index']
    distribution['class'] = Y_labels[0]

    # Testing data
    Y_test_labels = Y_test.map(lambda x: "Test Y" if x else "Test N")
    Y_test_labels = Y_test_labels.reset_index()
    distribution_test = pd.DataFrame()
    distribution_test['value'] = X_scaled_test_df[top_event]
    distribution_test = distribution_test.reset_index()
    del distribution_test['index']
    distribution_test['class'] = Y_test_labels[0]

    #distribution_both = pd.concat([distribution, distribution_test])

    return (distribution, distribution_test)

histograms = []
for idx in weights[:10]:
    event_name = X_scaled_df.columns[idx]
    data_training, data_test = get_histogram_data(idx)
    histogram_train = Histogram(data_training, background_fill_alpha=0.5, values='value', color='class', bins=20, title="Training samples for {} weight {:.2f}".format(event_name, abs_coefs[idx]), legend='top_right', label='class')

    histogram_test = Histogram(data_test, background_fill_alpha=0.5, values='value', color='class', bins=20, title="Test samples for {} weight {:.2f}".format(event_name, abs_coefs[idx]), legend='top_right', label='class')

    histograms.append((histogram_train, histogram_test))

env = Environment(loader=FileSystemLoader(os.path.join(sys.path[0], "templates")))
env.globals['folders'] = get_menu()
env.globals['config'] = config
env.globals['app'] = app
template = env.get_template('svmweights.html')

grid = gridplot([
    [histograms[0][0], histograms[0][1]],
    [histograms[1][0], histograms[1][1]],
    [histograms[2][0], histograms[2][1]],
    [histograms[3][0], histograms[3][1]],
    [histograms[4][0], histograms[4][1]],
    [histograms[5][0], histograms[5][1]],
    [histograms[6][0], histograms[6][1]],
    [histograms[7][0], histograms[7][1]],
    [histograms[8][0], histograms[8][1]],
    [histograms[9][0], histograms[9][1]]
], plot_width=650, plot_height=350)

curdoc().add_root(grid)
curdoc().title = "Distribution for high-weight SVM features"
curdoc().template = template
