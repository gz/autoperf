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
## clf = svm.CLASSIFIERS['linear']
## min_max_scaler = sklearn.preprocessing.MinMaxScaler()
## X_scaled = min_max_scaler.fit_transform(X)
## X_test_scaled = min_max_scaler.transform(X_test)
## model = clf.fit(X_scaled, Y)
## Y_pred = clf.predict(X_test_scaled)
## metrics = svm.get_svm_metrics(args, [app], Y, Y_test, Y_pred)
## source = ColumnDataSource(ColumnDataSource.from_df(pd.DataFrame([metrics])))
## X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
## X_scaled_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
##
## abs_coefs = np.abs(model.coef_[0])
## weight_idx_sorted = np.argsort( abs_coefs )
## weights = list(weight_idx_sorted)
## weights.reverse()

weights = [
    "MIN.L2_RQSTS.CODE_RD_MISS",
    "MIN.OFFCORE_REQUESTS.DEMAND_CODE_RD",
    "MIN.OFFCORE_RESPONSE.DEMAND_CODE_RD.LLC_HIT.ANY_RESPONSE",
    "MIN.L2_RQSTS.ALL_CODE_RD",
    "AVG.OFFCORE_RESPONSE.DEMAND_CODE_RD.LLC_HIT.ANY_RESPONSE",
    "MIN.BR_MISP_EXEC.TAKEN_RETURN_NEAR",
    "MIN.L2_TRANS.CODE_RD",
    "AVG.OFFCORE_REQUESTS.DEMAND_CODE_RD",
    "MIN.BR_INST_EXEC.TAKEN_INDIRECT_NEAR_CALL",
    "AVG.L2_RQSTS.CODE_RD_MISS",
    "AVG.L2_RQSTS.ALL_CODE_RD",
    "MAX.RESOURCE_STALLS.ROB",
    "MIN.L2_RQSTS.CODE_RD_HIT",
    "MAX.DSB_FILL.EXCEED_DSB_LINES",
    "AVG.DSB_FILL.EXCEED_DSB_LINES",
    "AVG.RESOURCE_STALLS.ROB",
    "AVG.MOVE_ELIMINATION.INT_NOT_ELIMINATED",
    "AVG.BR_MISP_EXEC.TAKEN_INDIRECT_NEAR_CALL",
    "MIN.RESOURCE_STALLS.ROB",
    "MAX.RS_EVENTS.EMPTY_CYCLES",
    "MAX.MOVE_ELIMINATION.INT_NOT_ELIMINATED",
    "MIN.MEM_UOPS_RETIRED.SPLIT_STORES",
    "MIN.BR_MISP_EXEC.TAKEN_INDIRECT_NEAR_CALL",
    "AVG.L2_TRANS.CODE_RD",
]

plots = []
for idx in weights[:15]:
    #event_name = X.columns[idx]
    event_name = idx

    yes = X[Y==True]
    no  = X[Y==False]
    yes = yes.reset_index()
    no = no.reset_index()

    name = "Training {} weight {:.2f}".format(event_name, 0.0)
    p_train = figure(plot_width=400, plot_height=400, title=name)
    p_train.line(yes.index.values, yes[event_name], line_width=2, color="red", legend="Interference")
    p_train.line(no.index.values, no[event_name], line_width=2, color="green", legend="No Interference")

    yes = X_test[Y_test==True]
    no  = X_test[Y_test==False]
    yes = yes.reset_index()
    no = no.reset_index()

    name = "Test {} weight {:.2f}".format(event_name, 0.0)
    p_test = figure(plot_width=400, plot_height=400, title=name)
    p_test.line(yes.index.values, yes[event_name], line_width=2, color="red", legend="Interference")
    p_test.line(no.index.values, no[event_name], line_width=2, color="green", legend="No Interference")

    plots.append( (p_train, p_test) )

env = Environment(loader=FileSystemLoader(os.path.join(sys.path[0], "templates")))
env.globals['folders'] = get_menu()
env.globals['config'] = config
env.globals['app'] = app
template = env.get_template('yesno.html')

grid = gridplot([
    [plots[0][0], plots[0][1]],
    [plots[1][0], plots[1][1]],
    [plots[2][0], plots[2][1]],
    [plots[3][0], plots[3][1]],
    [plots[4][0], plots[4][1]],
    [plots[5][0], plots[5][1]],
    [plots[6][0], plots[6][1]],
    [plots[7][0], plots[7][1]],
    [plots[8][0], plots[8][1]],
    [plots[9][0], plots[9][1]],
    [plots[10][0], plots[10][1]],
    [plots[11][0], plots[11][1]],
    [plots[12][0], plots[12][1]],
    [plots[13][0], plots[13][1]],
    [plots[14][0], plots[14][1]],
], plot_width=650, plot_height=350)

curdoc().add_root(grid)
curdoc().title = "Show overlaps for SVM features"
curdoc().template = template
