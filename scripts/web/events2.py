import pandas as pd
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import gridplot, column, row
from bokeh.charts import Line, show, output_file, defaults
from bokeh.models.widgets import PreText, Select
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show


df = pd.read_csv('matrix_X_uncore_shared.csv')

ticker = Select(value="XXX", options=list(df.columns.values), width=500)

defaults.plot_width = 450
defaults.plot_height = 400


source = ColumnDataSource(data=dict(x=[], y=[]))

#print ColumnDataSource.from_df(df['AVG.ARITH.FPU_DIV'])
#line.source.data = ColumnDataSource.from_df(df['AVG.ARITH.FPU_DIV'])
#ds = line.data

def ticker_change(attrname, old, new):
    print attrname, old, new
    fig.title = new
    source.data = dict(x=df.index.values, y=df[new])
ticker.on_change('value', ticker_change)

fig = figure(title='Event', width=1024, height=768)

line = fig.line('x', 'y', source=source, line_width=5)
widgets = column(ticker, fig)
curdoc().add_root(widgets)
curdoc().title = "Events"
