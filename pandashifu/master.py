import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, dash_table, dcc, html, Input, Output, State, callback

import pygwalker as pyg
import dash_dangerously_set_inner_html

import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import statsmodels.formula.api as smf
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from collections.abc import Iterable

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, f1_score

from .styles import theme_style, tab_style, tabs_style, buttons_style
from .styles import sidebar_tab_style, sidebar_tabs_style
from .styles import sidebar_style, content_style
from .styles import selected_color, unselected_color
from .plan import blueprint, PSNode
from .jade import export

from .ingredients import *
from .scroll import *


show = {'display': 'block'}
hide = {'display': 'none'}
sep = dcc.Markdown("---")


def data_frame(data, entire=True, extra_style=[]):

    info = f"{data.shape[0]} rows $\\times$ {data.shape[1]} columns\n"
    indexed_data = data.reset_index()
    index_col = list(set(indexed_data.columns) - set(data.columns))
    cstyle = [{'if': {'column_id': '_'.join(col)},
               'fontWeight': 'bold',
               'backgroundColor': '#FAFAFA'} for col in index_col]
    cstyle += extra_style
    columns = [{"name": col, "id": "_".join(col)}
               for col in indexed_data.columns]
    values = [{"_".join(col): val for col, val in row.items() }
              for row in indexed_data.to_dict('records')]
    view = dash_table.DataTable(values, columns,
                                page_size=10,
                                merge_duplicate_headers=True,
                                style_table={'overflowX': 'auto'},
                                style_header={'fontWeight': 'bold'},
                                style_cell={'fontSize': 12, 'height': 14, 'maxWidth': 150},
                                style_data_conditional=cstyle)

    if entire:
        return [dcc.Markdown('**Output data frame**:'),
                dcc.Markdown(info, mathjax=True),
                view]
    else:
        return [view]


def two_columns(left, right, id, style={}):

    sidebar = html.Div(left, id=f'{id}_sidebar', style=sidebar_style)
    content = html.Div(right, id=f'{id}_content', style=content_style)

    return html.Div([sidebar, content], id=id, style=style)

def new_node(figure, tabs, nodes, n_clicks, content, code, current, ntype='data'):

    if n_clicks is None:
        selected = {"points": [{"pointIndex": current['idx']}]}
    else:
        tabs = 'bp_tab'
        selected = {"points": [{"pointIndex": len(nodes)}]}

    name = content[0]
    idx = current['idx']
    if idx is not None and name is not None and n_clicks is not None:
        current_node = nodes[idx]
        next_node = current_node.grow(ntype, content, code=code)
        nodes.append(next_node)

        xp, yp = nodes[0].get_all_lines()
        figure['data'][0]['x'] = xp
        figure['data'][0]['y'] = yp
        xs = [n.pos[0] for n in nodes]
        ys = [n.pos[1] for n in nodes]

        is_visual = np.array([n.ntype == 'visual' for n in nodes])
        is_model = np.array([n.ntype == 'model' for n in nodes])
        num_nodes = len(nodes)
        marker_size = np.ones(num_nodes) * 30
        marker_size[is_visual] = 18
        marker_size[is_model] = 20
        marker_symbol = ['triangle-left' if n.ntype == 'model' else
                         'square' if n.ntype == 'visual' else
                         'circle' for n in nodes]

        colors = [unselected_color] * len(xs)
        colors[idx] = selected_color

        figure['data'][1]['x'] = xs
        figure['data'][1]['y'] = ys
        figure['data'][1]['marker']['color'] = colors
        figure['data'][1]['marker']['size'] = marker_size
        figure['data'][1]['marker']['symbol'] = marker_symbol
        figure['data'][1]['customdata'] = [n.content_label() for n in nodes]
    
    return figure, tabs, selected


############ Each operations #################
def op_none(current, nodes, id):

    # global output_name

    idx = current['idx']
    input_name, input_data = nodes[idx].content

    controls = html.Div([])

    table = html.Div(data_frame(input_data), id=f'{id}_table')

    code_string = f'```python\n{input_name}\n```'
    previews = html.Div([code_box(code_string, id), sep, table])

    return controls, previews


def select_columns(current, nodes, id):

    global output_name

    idx = current['idx']
    input_name, input_data = nodes[idx].content

    column_labels = list(index_labels(input_data.columns).index)

    name_out = sidebar_input('Output name:', 'text', f'{id}_name_out', placeholder="...")
    columns = sidebar_dropdown('Columns', column_labels, column_labels,
                               id=f'{id}_columns', multi=True)
    buttons = dbc.Row([std_button('Reset', id=f'{id}_reset_button'),
                       std_button('Save', id=f'{id}_add_button')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    controls = html.Div([name_out, sep,
                         columns, sep,
                         buttons], id=f'{id}_controls')

    table = html.Div(data_frame(input_data), id=f'{id}_table')
    code_string = f'```python\n{input_name}\n```'
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([code_box(code_string, id), error, sep, table])

    @callback(Output(f'{id}_table', 'children'),
              Output(f'{id}_code', 'children'),
              Input(f'{id}_columns', 'value'),
              Input(f'{id}_name_out', 'value'),
              State('current', 'data'))
    def update_df_columns(columns, output_name, current):

        global output_data

        idx = current['idx']
        input_name, input_data = nodes[idx].content

        column_labels = index_labels(input_data.columns)
        output_data = input_data.loc[:, column_labels[columns]]

        code = columns_code(input_data, output_name, input_name, columns)

        return data_frame(output_data), code

    @callback(Output(f'{id}_add_button', 'disabled'),
              Output(f'{id}_error', 'children'),
              Input(f'{id}_name_out', 'value'),
              Input(f'{id}_columns', 'value'))
    def update_save_disabled(name, columns):

        all_names = [node.content[0] for node in nodes]
        name_conflict = name in all_names
        if len(columns) == 0 :
            return True, ''
        elif isinvalid(name) :
            return True, ''
        elif name_conflict:
            return True, f'\n\n**`Error`**`: Name "{name}" has been used to define another dataset.`'
        else:
            return False, ''

    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_add_button', 'n_clicks'),
              State('blueprint', 'figure'),
              State(f'{id}_name_out', 'value'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_df_columns_add(n_clicks, figure, name, tabs, current, code_markdown):

        global output_data

        code = code_markdown.replace('```python\n', '').replace('\n```', '')
        content = name, output_data

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'data')
        #return add_node(n_clicks, figure, name, tabs, current, nodes, output_data, code)


    @callback(Output(f'{id}_columns', 'value'),
              Input(f'{id}_reset_button', 'n_clicks'),
              State(f'{id}_columns', 'options'),
              prevent_initial_call=True)
    def update_df_columns_reset_button(n_clicks, options):

        return options

    return controls, previews


def filter_rows(current, nodes, id):

    # global output_name

    idx = current['idx']
    input_name, input_data = nodes[idx].content

    column_labels = list(index_labels(input_data.columns).index)
    filter_list = [{'column': None, 'values': [], 'range': None}]

    name_out = sidebar_input('Output name:', 'text', f'{id}_name_out', placeholder="...")

    ftype = sidebar_dropdown('Filter type', ['Values', 'Range'], None, id=f'{id}_type')
    column = sidebar_dropdown('Filter on column', column_labels, None, id=f'{id}_column')
    expr = sidebar_expr('Values', id=f'{id}_expr')
    #frange = sidebar_range('Range', 0, 1, 0.001, id=f'{id}_range')
    frange = html.Div([sidebar_input('Min value:', 'number', id=f'{id}_min'),
                       sidebar_input('Max value:', 'number', id=f'{id}_max')], id=f'{id}_range_div')
    add = dbc.Row([std_button('Add Filter', id=f'{id}_add', width=120)], justify='end',
                  style={'margin-left': 3, 'margin-right': 3, 'margin-top': 0, 'margin-bottom': 5})
    reset = sidebar_switch('Reset index', 'no/yes', id=f'{id}_reset')

    store = dcc.Store('filter_list', data=filter_list)
    buttons = dbc.Row([std_button('Reset', id=f'{id}_reset_button'),
                       std_button('Save', id=f'{id}_add_button')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    controls = html.Div([store, name_out, sep,
                         ftype, column, expr, frange, add, reset, sep,
                         buttons], id=f'{id}_controls')

    table = html.Div(data_frame(input_data), id=f'{id}_table')
    code_string = f'```python\n{input_name}\n```'
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([code_box(code_string, id), error, sep, table])

    @callback(Output(f'{id}_column_div', 'style'),
              Output(f'{id}_column', 'value'),
              Output(f'{id}_expr_div', 'style'),
              Output(f'{id}_range_div', 'style'),
              Output(f'{id}_column', 'options'),
              Input(f'{id}_type', 'value'),
              State('current', 'data'))
    def update_filter_type(ftype, current):

        idx = current['idx']
        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)

        if ftype is None:
            return hide, None, hide, hide, []
        elif ftype == 'Values':
            return show, None, show, hide, list(column_labels.keys())
        elif ftype == 'Range':
            num_column_labels = index_labels(input_data.select_dtypes(include='number').columns)
            return show, None, hide, show, list(num_column_labels.keys())

    @callback(Output(f'{id}_min', 'min'),
              Output(f'{id}_min', 'max'),
              Output(f'{id}_min', 'step'),
              Output(f'{id}_min', 'value'),
              Output(f'{id}_max', 'min'),
              Output(f'{id}_max', 'max'),
              Output(f'{id}_max', 'step'),
              Output(f'{id}_max', 'value'),
              Input(f'{id}_column', 'value'),
              State(f'{id}_range_div', 'style'),
              State('current', 'data'))
    def update_filter_range(column, style, current):

        if style['display'] == 'none' or column is None:
            return (None,) * 8
        else:
            idx = current['idx']
            input_name, input_data = nodes[idx].content
            column_labels = index_labels(input_data.columns)
            filter_on = input_data[column_labels[column]].dropna()
            fmin, fmax = filter_on.min(), filter_on.max()
            if (filter_on.astype(int) == filter_on).all():
                step = 1
            else:
                #step = 10 ** np.floor(np.log10((fmax - fmin)/1000))
                step = np.minimum(1, 10 ** np.floor(np.log10((fmax - fmin)/1000)))
                multiplier = 10 ** np.log10(step)
                fmin = np.floor(fmin/multiplier) * multiplier
                fmax = np.ceil(fmax/multiplier) * multiplier
            return fmin, fmax, step, fmin, fmin, fmax, step, fmax

    @callback(Output(f'{id}_expr', 'value'),
              Input(f'{id}_column', 'value'))
    def update_filter_value(column):

        return None

    @callback(Output(f'{id}_table', 'children', allow_duplicate=True),
              Output(f'{id}_code', 'children', allow_duplicate=True),
              Output('filter_list', 'data', allow_duplicate=True),
              Output(f'{id}_add', 'disabled'),
              Output(f'{id}_add_button', 'disabled'),
              Output(f'{id}_error', 'children', allow_duplicate=True),
              Input(f'{id}_expr', 'value'),
              Input(f'{id}_min', 'value'),
              Input(f'{id}_max', 'value'),
              Input(f'{id}_reset', 'value'),
              # Input(f'{id}_open', 'value'),
              Input(f'{id}_name_out', 'value'),
              State(f'{id}_column', 'value'),
              State(f'{id}_table', 'children'),
              State('filter_list', 'data'),
              State(f'current', 'data'),
              prevent_initial_call=True)
    def update_filter(expr, fmin, fmax, reset, output_name, column, table, flist, current):

        global output_data

        idx = current['idx']
        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)

        error_string = ''
        disable_add = True
        disable_add_button = True
        if column is not None:
            filter_on = input_data[column_labels[column]]
            current_filter = flist[-1]
            if fmin is not None and fmax is not None:
                current_filter['column'] =  column_labels[column]
                current_filter['values'] = []
                current_filter['range'] = [fmin, fmax]
                disable_add = False
                disable_add_button = False
            elif expr is not None:
                if expr.strip() == '':
                    #return data_frame(input_data), '', ''
                    current_filter['column'] =  None
                    current_filter['values'] = []
                    current_filter['range'] = None
                else:
                    try:
                        values = eval(expr)
                        current_filter['column'] = column_labels[column]
                        current_filter['range'] = None
                        if isinstance(values, str) or not isinstance(values, Iterable):
                            #output_data = input_data.loc[filter_on == values]
                            current_filter['values'] = [values]
                        elif isinstance(values, Iterable):
                            #output_data = input_data.loc[filter_on.isin(tuple(values))]
                            current_filter['values'] = list(values)
                        #return data_frame(output_data), '', ''
                        disable_add = False
                        disable_add_button = False
                    except Exception as err:
                        # return table, '', f'\n\n**`Error`**`: {err}`'
                        error_string = f'\n\n**`Error`**`: {err}`'
        cond = np.array([True] * input_data.shape[0])
        for f in flist:
            if f['column'] is None:
                continue
            filter_on = input_data[column_labels[f['column']]]
            if len(f['values']) == 1:
                cond = cond & (filter_on == f['values'][0])
            elif len(f['values']) > 1:
                cond = cond & (filter_on.isin(f['values']))
            if f['range'] is not None:
                fmin, fmax = f['range']
                cond = cond & (filter_on >= fmin) & (filter_on <= fmax)
            disable_add_button = False

        output_data = input_data.loc[cond]
        if len(reset) > 0:
            output_data = output_data.reset_index(drop=True)

        all_names = [node.content[0] for node in nodes]
        if isinvalid(output_name):
            disable_add_button = True
        elif output_name in all_names:
            disable_add_button = True
            error_string = f'\n\n**`Error`**`: Name "{output_name}" has been used to define another dataset.`'

        code = filter_code(output_name, input_name, flist, len(reset)>0)

        return data_frame(output_data), code, flist, disable_add, disable_add_button, error_string

    @callback(Output(f'{id}_type', 'value'),
              Output('filter_list', 'data', allow_duplicate=True),
              Input(f'{id}_add', 'n_clicks'),
              State('filter_list', 'data'),
              prevent_initial_call=True)
    def update_add_filter(n_clicks, flist):

        flist.append({'column': None, 'values': [], 'range': None})

        return None, flist

    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_add_button', 'n_clicks'),
              State('blueprint', 'figure'),
              State(f'{id}_name_out', 'value'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_df_filter_add(n_clicks, figure, name, tabs, current, code_markdown):

        global output_data

        code = code_markdown.replace('```python\n', '').replace('\n```', '')
        content = name, output_data

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'data')

        #return add_node(n_clicks, figure, name, tabs, current, nodes, output_data, code)

    @callback(Output(f'{id}_type', 'value', allow_duplicate=True),
              Output('filter_list', 'data', allow_duplicate=True),
              Input(f'{id}_reset_button', 'n_clicks'),
              prevent_initial_call=True)
    def update_df_filter_reset_button(n_clicks):

        return None, [{'column': None, 'values': [], 'range': None}]

    return controls, previews


def treat_na(current, nodes, id):

    global output_data

    idx = current['idx']
    input_name, input_data = nodes[idx].content
    column_labels = list(index_labels(input_data.columns).index)

    name_out = sidebar_input('Output name:', 'text', f'{id}_name_out', placeholder="...")
    treat_method = sidebar_dropdown('Method',
                                    ['drop', 'fill'], None, id=f'{id}_method')
    all_switch = sidebar_switch('All columns', 'no/yes', id=f'{id}_all_switch')
    columns = sidebar_dropdown('Columns',  column_labels, None,
                               id=f'{id}_columns', multi=True)
    fill_value = sidebar_expr('Value to fill', id=f'{id}_value')

    buttons = dbc.Row([std_button('Reset', id=f'{id}_reset_button'),
                       std_button('Save', id=f'{id}_add_button')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    controls = html.Div([name_out, sep,
                         treat_method, all_switch, columns, fill_value, sep,
                         buttons], id=f'{id}_controls')

    table = html.Div(data_frame(input_data), id=f'{id}_table')
    code_string = f'```python\n{input_name}\n```'
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([code_box(code_string, id), error, sep, table])

    @callback(Output(f'{id}_columns_div', 'style'),
              Output(f'{id}_columns', 'options'),
              Input(f'{id}_all_switch', 'value'),
              State('current', 'data'))
    def update_na_columns(all_switch, current):

        if len(all_switch) == 0:
            idx = current['idx']
            input_data = nodes[idx].content[1]
            has_na = (input_data.isnull().sum() > 0)
            options = list(index_labels(input_data.columns[has_na]).index)

            return show, options
        else:
            return hide, []
    
    @callback(Output(f'{id}_value_div', 'style'),
              Input(f'{id}_method', 'value'),
              State('current', 'data'))
    def update_fill_value(method, current):

        if method == 'fill':
            return show
        else:
            return hide
    
    @callback(Output(f'{id}_table', 'children'),
              Output(f'{id}_code', 'children'),
              Output(f'{id}_error', 'children'),
              Output(f'{id}_add_button', 'disabled'),
              Input(f'{id}_name_out', 'value'),
              Input(f'{id}_method', 'value'),
              Input(f'{id}_all_switch', 'value'),
              Input(f'{id}_columns', 'value'),
              Input(f'{id}_value', 'value'),
              State('current', 'data'))
    def update_treat_na_table(output_name, method, all_switch,
                              columns, values, current):
        
        global output_data

        idx = current['idx']
        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)

        error_string = ''
        disable = True
        output_data = input_data
        if len(all_switch) > 0:
            if method == 'drop':
                output_data = input_data.dropna()
                disable = False
            elif method == 'fill':
                if values is not None:
                    try:
                        output_data = input_data.fillna(eval(values))
                        disable = False
                    except Exception as err:
                        error_string = f'\n\n**`Error`**`: {err}`'
        else:
            if columns is not None:
                has_na = input_data[list(column_labels[columns])].isnull().any(axis=1)
                if method == 'drop':
                    output_data = input_data.loc[~has_na]
                    disable = False
                elif method == 'fill':
                    output_data = input_data.copy()
                    if values is not None:
                        try:
                            subset = output_data.loc[:, list(column_labels[columns])].fillna(eval(values))
                            output_data.fillna(subset, inplace=True)
                            disable = False
                        except Exception as err:
                            error_string = f'\n\n**`Error`**`: {err}`'
        
        all_names = [node.content[0] for node in nodes]
        name_conflict = output_name in all_names
        disable = disable or isinvalid(output_name) or output_name == '' or name_conflict

        code = treat_na_code(output_name, input_name, column_labels,
                             method, all_switch, columns, values)
        
        return data_frame(output_data), code, error_string, disable
    
    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_add_button', 'n_clicks'),
              State('blueprint', 'figure'),
              State(f'{id}_name_out', 'value'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_treat_na_add(n_clicks, figure, name, tabs, current, code_markdown):

        global output_data

        code = code_markdown.replace('```python\n', '').replace('\n```', '')
        content = name, output_data

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'data')
    
    @callback(Output(f'{id}_method', 'value'),
              Output(f'{id}_all_switch', 'value'),
              Output(f'{id}_columns', 'value'),
              Output(f'{id}_value', 'value'),
              Input(f'{id}_reset_button', 'n_clicks'))
    def update_treat_na_reset_button(n_clicks):

        return None, [], None, None

    return controls, previews


def sort_rows(current, nodes, id):

    global output_data

    idx = current['idx']
    input_name, input_data = nodes[idx].content

    column_labels = list(index_labels(input_data.columns).index)
    sort_list = [{'column': None, 'ascending': True}]

    name_out = sidebar_input('Output name:', 'text', f'{id}_name_out', placeholder="...")
    column = sidebar_dropdown('By column', column_labels, None, id=f'{id}_column')
    order = sidebar_switch('Descending', 'no/yes', id=f'{id}_order')
    add = dbc.Row([std_button('Add Criterion', id=f'{id}_add', width=120)],justify='end',
                  style={'margin-left': 3, 'margin-right': 3, 'margin-top': 0, 'margin-bottom': 5})
    reset = sidebar_switch('Reset index', 'no/yes', id=f'{id}_reset')

    store = dcc.Store('sort_list', data=sort_list)
    buttons = dbc.Row([std_button('Reset', id=f'{id}_reset_button'),
                       std_button('Save', id=f'{id}_add_button')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    controls = html.Div([store, name_out, sep,
                         column, order, add, reset, sep,
                         buttons], id=f'{id}_controls')

    table = html.Div(data_frame(input_data), id=f'{id}_table')
    code_string = f'```python\n{input_name}\n```'
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([code_box(code_string, id), error, sep, table])

    @callback(Output(f'{id}_table', 'children'),
              Output(f'{id}_code', 'children'),
              Output('sort_list', 'data'),
              Output(f'{id}_add', 'disabled'),
              Output(f'{id}_add_button', 'disabled'),
              Output(f'{id}_error', 'children'),
              Input(f'{id}_column', 'value'),
              Input(f'{id}_order', 'value'),
              Input(f'{id}_reset', 'value'),
              Input(f'{id}_name_out', 'value'),
              State('current', 'data'),
              State('sort_list', 'data'))
    def update_sort(column, order, reset, output_name, current, sort_list):

        global output_data

        idx = current['idx']
        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)

        if column is not None:
            ascending = False if len(order) > 0 else True
            current_sort = sort_list[-1]
            current_sort['column'] = column_labels[column]
            current_sort['ascending'] = ascending

        columns = []
        ascendings = []
        for s in sort_list:
            if s['column'] is None:
                continue
            new_columns = s['column']
            if isinstance(new_columns, list):
                new_columns = tuple(new_columns)
            columns.append(new_columns)
            ascendings.append(s['ascending'])

        if len(columns) > 0:
            output_data = input_data.sort_values(columns[::-1], ascending=ascendings[::-1])
            if len(reset) > 0:
                output_data = output_data.reset_index(drop=True)
        else:
            output_data = input_data

        code = sort_code(output_name, input_name, columns, ascendings, len(reset)>0)

        disable_add = column is None
        all_names = [node.content[0] for node in nodes]
        if isinvalid(output_name) or len(columns) == 0:
            disable_add_button = True
            error = ''
        elif output_name in all_names:
            disable_add_button = True
            error = f'\n\n**`Error`**`: Name "{output_name}" has been used to define another dataset.`'
        else:
            disable_add_button = False
            error = ''

        return data_frame(output_data), code, sort_list, disable_add, disable_add_button, error

    @callback(Output(f'{id}_column', 'value'),
              Output(f'{id}_order', 'value'),
              Output('sort_list', 'data', allow_duplicate=True),
              # Output('debug_info', 'children', allow_duplicate=True),
              Input(f'{id}_add', 'n_clicks'),
              State('sort_list', 'data'),
              prevent_initial_call=True)
    def update_add_sort(n_clicks, sort_list):

        sort_list.append({'column': None, 'ascending': True})

        return None, [], sort_list

    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_add_button', 'n_clicks'),
              State('blueprint', 'figure'),
              State(f'{id}_name_out', 'value'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_df_sort_add(n_clicks, figure, name, tabs, current, code_markdown):

        global output_data

        code = code_markdown.replace('```python\n', '').replace('\n```', '')
        content = name, output_data

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'data')

        #return add_node(n_clicks, figure, name, tabs, current, nodes, output_data, code)

    @callback(Output(f'{id}_column', 'value', allow_duplicate=True),
              Output(f'{id}_order', 'value', allow_duplicate=True),
              Output(f'{id}_reset', 'value', allow_duplicate=True),
              Output('sort_list', 'data', allow_duplicate=True),
              Input(f'{id}_reset_button', 'n_clicks'),
              prevent_initial_call=True)
    def update_df_sort_reset_button(n_clicks):

        return None, [], [], [{'column': None, 'ascending': True}]


    return controls, previews


def grouped(current, nodes, id):

    global output_data

    idx = current['idx']
    input_name, input_data = nodes[idx].content

    column_labels = list(index_labels(input_data.columns).index)

    name_out = sidebar_input('Output name:', 'text', f'{id}_name_out', placeholder="...")
    by_columns = sidebar_dropdown('Group by', column_labels, [],
                                  id=f'{id}_by', multi=True)
    view_columns = sidebar_dropdown('View on', column_labels, [],
                                    id=f'{id}_view', multi=True)
    calc = ['count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var']
    apply = sidebar_dropdown('Apply', calc, None,
                             id=f'{id}_apply', multi=True)
    reset = sidebar_switch('Reset index', 'no/yes', id=f'{id}_reset')
    buttons = dbc.Row([std_button('Reset', id=f'{id}_reset_button'),
                       std_button('Save', id=f'{id}_add_button')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    controls = html.Div([name_out, sep,
                         by_columns, view_columns, apply, reset, sep,
                         buttons], id=f'{id}_controls')

    table = html.Div(data_frame(input_data), id=f'{id}_table')
    #error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red'})
    code_string = f'```python\n{input_name}\n```'
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([code_box(code_string, id), error, sep, table])

    @callback(Output(f'{id}_table', 'children'),
              Output(f'{id}_code', 'children'),
              Output(f'{id}_add_button', 'disabled'),
              Output(f'{id}_error', 'children'),
              Input(f'{id}_by', 'value'),
              Input(f'{id}_view', 'value'),
              Input(f'{id}_apply', 'value'),
              Input(f'{id}_reset', 'value'),
              Input(f'{id}_name_out', 'value'),
              State('current', 'data'),
              State(f'{id}_table', 'children'))
    def update_groupby(by, view, apply, reset, output_name, current, table):

        global output_data

        idx = current['idx']
        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)

        if by and view and apply:
            column_labels = index_labels(input_data.columns)
            columns_by = list(column_labels[by])
            columns_view = list(column_labels[view])
            code = grouped_code(output_name, input_name, columns_by, columns_view, apply, len(reset)>0)
            try:
                output_data = input_data.groupby(columns_by)[columns_view].agg(apply)
                if len(reset) > 0:
                    output_data = output_data.reset_index()
                all_names = [node.content[0] for node in nodes]
                name_conflict = output_name in all_names
                disable_add_button = isinvalid(output_name) or output_name == '' or name_conflict
                error = '' if not name_conflict else f'\n\n**`Error`**`: Name "{output_name}" has been used to define another dataset.`'
                return data_frame(output_data), code, disable_add_button, error
            except Exception as err:
                return table, code, True, f'\n\n**`Error`**`: {err}`'
        else:
            left = '' if output_name is None or output_name == '' else f'{output_name} = '
            return table, f'```python\n{left}{input_name}\n```', True, ''

    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              #Output('blueprint', 'selectedData', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              #Output(f'{id}_error', 'children', allow_duplicate=True),
              Input(f'{id}_add_button', 'n_clicks'),
              State('blueprint', 'figure'),
              State(f'{id}_name_out', 'value'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_df_columns_add(n_clicks, figure, name, tabs, current, code_markdown):

        global output_data

        code = code_markdown.replace('```python\n', '').replace('\n```', '')
        content = name, output_data

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'data')
        #return add_node(n_clicks, figure, name, tabs, current, nodes, output_data, code)

    @callback(Output(f'{id}_table', 'children', allow_duplicate=True),
              Output(f'{id}_by', 'value'),
              Output(f'{id}_view', 'value'),
              Output(f'{id}_apply', 'value'),
              Output(f'{id}_reset', 'value'),
              Input(f'{id}_reset_button', 'n_clicks'),
              State('current', 'data'),
              prevent_initial_call=True)
    def update_df_grouped_reset_button(n_clicks, current):

        idx = current['idx']
        input_name, input_data = nodes[idx].content

        return data_frame(input_data), None, None, None, []

    return controls, previews


def pivot(current, nodes, id):

    global output_data

    idx = current['idx']
    input_name, input_data = nodes[idx].content

    column_labels = list(index_labels(input_data.columns).index)

    name_out = sidebar_input('Output name:', 'text', f'{id}_name_out', placeholder="...")
    values = sidebar_dropdown('Values', column_labels, [],
                              id=f'{id}_values', multi=True)
    index = sidebar_dropdown('Index', column_labels, [],
                             id=f'{id}_index', multi=True)
    columns = sidebar_dropdown('Columns', column_labels, [],
                               id=f'{id}_columns', multi=True)
    calc = ['count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var']
    agg = sidebar_dropdown('Aggregation', calc, None,
                           id=f'{id}_agg', multi=True)
    reset = sidebar_switch('Reset index', 'no/yes', id=f'{id}_reset')
    buttons = dbc.Row([std_button('Reset', id=f'{id}_reset_button'),
                       std_button('Save', id=f'{id}_add_button')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    controls = html.Div([name_out, sep,
                         values, index, columns, agg, reset, sep,
                         buttons], id=f'{id}_controls')

    table = html.Div(data_frame(input_data), id=f'{id}_table')
    code_string = f'```python\n{input_name}\n```'
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([code_box(code_string, id), error, sep, table])

    @callback(Output(f'{id}_table', 'children'),
              Output(f'{id}_code', 'children'),
              Output(f'{id}_add_button', 'disabled'),
              Output(f'{id}_error', 'children'),
              Input(f'{id}_values', 'value'),
              Input(f'{id}_index', 'value'),
              Input(f'{id}_columns', 'value'),
              Input(f'{id}_agg', 'value'),
              Input(f'{id}_reset', 'value'),
              Input(f'{id}_name_out', 'value'),
              State('current', 'data'),
              State(f'{id}_table', 'children'))
    def update_pivot(values, index, columns, agg, reset, output_name, current, table):

        global output_data

        idx = current['idx']
        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)

        if values and index and columns and agg:
            column_labels = index_labels(input_data.columns)
            value_cols = list(column_labels[values])
            index_cols = list(column_labels[index])
            column_cols = list(column_labels[columns])
            code = pivot_code(output_name, input_name, value_cols, index_cols, column_cols, agg, len(reset)>0)
            try:
                output_data = input_data.pivot_table(value_cols, index_cols, column_cols, agg)
                if len(reset) > 0:
                    output_data = output_data.reset_index()
                all_names = [node.content[0] for node in nodes]
                name_conflict = output_name in all_names
                disable_add_button = isinvalid(output_name) or output_name == '' or name_conflict
                error = '' if not name_conflict else f'\n\n**`Error`**`: Name "{output_name}" has been used to define another dataset.`'
                return data_frame(output_data), code, disable_add_button, error
            except Exception as err:
                return table, code, True, f'\n\n**`Error`**`: {err}`'
        else:
            left = '' if output_name is None or output_name == '' else f'{output_name} = '
            return table, f'```python\n{left}{input_name}\n```', True, ''

    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_add_button', 'n_clicks'),
              State('blueprint', 'figure'),
              State(f'{id}_name_out', 'value'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_df_pivot_add(n_clicks, figure, name, tabs, current, code_markdown):

        global output_data

        code = code_markdown.replace('```python\n', '').replace('\n```', '')
        content = name, output_data
        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'data')

        # return add_node(n_clicks, figure, name, tabs, current, nodes, output_data, code)

    @callback(Output(f'{id}_table', 'children', allow_duplicate=True),
              Output(f'{id}_values', 'value'),
              Output(f'{id}_index', 'value'),
              Output(f'{id}_columns', 'value'),
              Output(f'{id}_agg', 'value'),
              Output(f'{id}_reset', 'value'),
              Input(f'{id}_reset_button', 'n_clicks'),
              State('current', 'data'),
              prevent_initial_call=True)
    def update_df_pivot_reset_button(n_clicks, current):

        idx = current['idx']
        input_name, input_data = nodes[idx].content

        return data_frame(input_data), None, None, None, None, []

    return controls, previews


def add_column(current, nodes, id):

    global output_data

    idx = current['idx']
    input_name, input_data = nodes[idx].content

    column_labels = list(index_labels(input_data.columns).index)

    name_out = sidebar_input('Output name:', 'text', f'{id}_name_out', placeholder="...")
    new_column = sidebar_input('New column:', 'text', f'{id}_new_column', placeholder="...")
    type_expr = sidebar_dropdown('Expression type', 
                                 ['type conversion', 'arithmetic operations',
                                  'string operations', 'to datetime', 'to dummies'],
                                 None, id=f'{id}_type_expr')
    from_column = sidebar_dropdown('From column(s)', column_labels, None, id=f'{id}_from_column')
    exprs = sidebar_expr('Expressions', id=f'{id}_expr')
    new_dtype = sidebar_expr('Date type', id=f'{id}_dtype')
    time_format = sidebar_expr('Time format', id=f'{id}_time_format')
    drop_first_switch = sidebar_switch('Drop first', 'no/yes', id=f'{id}_drop_first_switch')
    buttons = dbc.Row([std_button('Reset', id=f'{id}_reset_button'),
                       std_button('Save', id=f'{id}_add_button')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    controls = html.Div([name_out, sep,
                         new_column, type_expr, from_column,
                         exprs, new_dtype, time_format, drop_first_switch, sep,
                         buttons], id=f'{id}_controls')

    table = html.Div(data_frame(input_data), id=f'{id}_table')
    code_string = f'```python\n{input_name}\n```'
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([code_box(code_string, id), error, sep, table])

    @callback(Output(f'{id}_from_column', 'multi'),
              Output(f'{id}_from_column', 'options'),
              Output(f'{id}_expr_div', 'style'),
              Output(f'{id}_dtype_div', 'style'),
              Output(f'{id}_time_format_div', 'style'),
              Output(f'{id}_drop_first_switch_div', 'style'),
              Input(f'{id}_type_expr', 'value'),
              State('current', 'data'))
    def update_from_column(etype, current):

        idx = current['idx']
        input_data = nodes[idx].content[1]
        all_columns = index_labels(input_data.columns)
        column_labels = list(all_columns.index)
        is_number = input_data.apply(is_numeric_dtype, axis=0).values
        
        if etype == 'arithmetic operations':
            multi = True
        else:
            multi = False

        if etype == 'arithmetic operations':
            options = list(all_columns.index[is_number])
        elif etype in ['string operations', 'to datetime', 'to dummies']:
            options = list(all_columns.index[~is_number])
        else:
            options = column_labels

        if etype is None:
            display = hide, hide, hide, hide
        elif etype == 'to dummies':
            display = hide, hide, hide, show
        elif etype == 'to datetime':
            display = hide, hide, show, hide
        elif etype == 'type conversion':
            display = hide, show, hide, hide
        else:
            display = show, hide, hide, hide

        return (multi, options) + display
    
    @callback(Output(f'{id}_expr', 'value'),
              Input(f'{id}_type_expr', 'value'),
              Input(f'{id}_from_column', 'value'),
              State(f'{id}_expr', 'value'),
              State('current', 'data'))
    def update_arithmetic_expr(etype, from_column, expr, current):

        if etype == 'arithmetic operations':
            if not from_column:
                return expr
            idx = current['idx']
            input_name, input_data = nodes[idx].content
            column_labels = index_labels(input_data.columns)
            terms = [f'{input_name}[{column_labels[c].__repr__()}]' for c in from_column]
            return ' + '.join(terms)
        else:
            return expr

    @callback(Output(f'{id}_table', 'children'),
              Output(f'{id}_code', 'children'),
              Output(f'{id}_add_button', 'disabled'),
              Output(f'{id}_error', 'children'),
              Input(f'{id}_name_out', 'value'),
              Input(f'{id}_new_column', 'value'),
              Input(f'{id}_type_expr', 'value'),
              Input(f'{id}_from_column', 'value'),
              Input(f'{id}_expr', 'value'),
              Input(f'{id}_dtype', 'value'),
              Input(f'{id}_time_format', 'value'),
              Input(f'{id}_drop_first_switch', 'value'),
              State('current', 'data'))
    def update_add_column_table(name_out, new_column, etype, from_column,
                                exprs, dtype, time_format, drop_first, current):
        
        global output_data

        idx = current['idx']
        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)

        output_data = input_data.copy()
        error_string = ''
        disable = True
        if etype == 'type conversion':
            if from_column is not None and new_column is not None and dtype is not None:
                try:
                    col = input_data[column_labels[from_column]].astype(eval(dtype))
                    output_data[new_column] = col
                    disable = False
                except Exception as err:
                    error_string = f'\n\n**`Error`**`: {err}`'
        elif etype == 'arithmetic operations':
            if from_column is not None and new_column is not None and exprs is not None:
                try:
                    formula = exprs.replace(f'{input_name}', 'input_data')
                    output_data[new_column] = eval(formula)
                    disable = False
                except Exception as err:
                    error_string = f'\n\n**`Error`**`: {err}`'
        elif etype == 'string operations':
            if from_column is not None and new_column is not None and exprs is not None:
                try:
                    col = eval(f'input_data[column_labels[from_column]].str{exprs}')
                    output_data[new_column] = col
                    disable = False
                except Exception as err:
                    error_string = f'\n\n**`Error`**`: {err}`'
        elif etype == 'to datetime':
            if from_column is not None and new_column is not None and time_format is not None:
                try:
                    col = pd.to_datetime(input_data[column_labels[from_column]], format=time_format)
                    output_data[new_column] = col
                    disable = False
                except Exception as err:
                    error_string = f'\n\n**`Error`**`: {err}`'
        elif etype == 'to dummies':
            if from_column is not None and new_column is not None:
                try:
                    cols = pd.get_dummies(input_data[column_labels[from_column]],
                                          drop_first=len(drop_first)>0, dtype=int)
                    dummy_columns = [f'{new_column}_{each}' for each in cols.columns]
                    output_data[dummy_columns] = cols.values
                    disable = False
                except Exception as err:
                    error_string = f'\n\n**`Error`**`: {err}`'
        all_names = [node.content[0] for node in nodes]
        name_conflict = name_out in all_names
        disable = disable or isinvalid(name_out) or name_out == '' or name_conflict
        
        code_string = add_colum_code(name_out, input_name, column_labels,
                                     new_column, etype, from_column,
                                     exprs, dtype, time_format, drop_first)
        
        return data_frame(output_data), code_string, disable, error_string
    
    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_add_button', 'n_clicks'),
              State('blueprint', 'figure'),
              State(f'{id}_name_out', 'value'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_df_pivot_add(n_clicks, figure, name, tabs, current, code_markdown):

        global output_data

        code = code_markdown.replace('```python\n', '').replace('\n```', '')
        content = name, output_data
        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'data')


    return controls, previews


########### Visualization ####################
def fig_controls(id, more=False):

    num_style = {'width': 90,
                 'height': 30,
                 'margin-left': 0,
                 'display': 'inline-block'}
    x_style = style={'width': 30,
                     'height': 30,
                     'margin-left': 0,
                     'text-align': 'center',
                     'display': 'inline-block'}

    if more:
        others = [sidebar_dropdown('Legend location',
                                   ['upper left', 'upper right', 'lower left', 'lower right'], 'upper left',
                                   id=f'{id}_legpos'),
                  sidebar_inline_expr('X-label', id=f'{id}_xlabel'),
                  sidebar_inline_expr('Y-label', id=f'{id}_ylabel'), sep]
    else:
        others = []
    

    configs = [sidebar_input('Rotate xticks', 'number', id=f'{id}_xrotate',
                             min=-90, max=90, step=5, value=0),
               sidebar_switch('Grid', 'off/on', f'{id}_grid_switch'),
               dcc.Markdown('Figure size:', style={'margin-left': 5, 'height': 30}),
               dbc.Input(type='number', id=f'{id}_figure_width',
                         min=150, max=750, value=600, step=5, style=num_style),
               dcc.Markdown(' x ', style=x_style),
               dbc.Input(type='number', id=f'{id}_figure_height',
                         min=150, max=750, value=400, step=5, style=num_style)]

    return html.Div(others + configs, id=f'{id}_config')

def visual_tabs(id, style={}):

    all_tabs = [dcc.Tab(id=f'{id}_plot_tab', value=f'plot_tab', label='Plot',
                        style=sidebar_tab_style, selected_style=sidebar_tab_style),
                dcc.Tab(id=f'{id}_fig_tab', value='fig_tab', label='Figure',
                        style=sidebar_tab_style, selected_style=sidebar_tab_style)]
    
    @callback(Output(f'{id}_plot_controls', 'style'),
              Output(f'{id}_config', 'style'),
              Input(f'{id}_tabs', 'value'))
    def update_visual_tabs(tab):

        if tab == 'plot_tab':
            return show, hide
        else:
            return hide, show

    return html.Div(dcc.Tabs(all_tabs, id=f'{id}_tabs', value='plot_tab', style=sidebar_tabs_style),
                    id=f'{id}_tabs_div', style=style)

def univar_visual(current, nodes, id):

    global output_data

    idx = current['idx']
    input_name, input_data = nodes[idx].content

    column_labels = list(index_labels(input_data.columns).index)

    plot_type = sidebar_dropdown('Type of plot',
                                 ['numerical results', 'value counts', 'hist', 'kde', 'boxplot'],
                                 'numerical results',
                                 id=f'{id}_ptype')
    uni_column = sidebar_dropdown('Selected column', column_labels, None,
                                 id=f'{id}_column')
    horizontal = html.Div([sidebar_switch('Horizontal', 'no/yes', f'{id}_horizontal')],
                          id=f'{id}_horizontal_comp')
    density = html.Div(sidebar_switch('Density', 'no/yes', f'{id}_density'),
                       id=f'{id}_density_comp')
    bar_width = html.Div(sidebar_input('Bar width', 'number', f'{id}_bar_width',
                                       min=0.1, max=1.0, value=0.75, step=0.05), id=f'{id}_bar_width_comp')
    num_bins = html.Div(sidebar_input('Bins', 'number', f'{id}_num_bins',
                                      min=5, max=80, value=10, step=1), id=f'{id}_num_bins_comp')
    plot_color = html.Div(sidebar_input('Color', 'color', f'{id}_color', value='#1f77b4'),
                          id=f'{id}_color_comp')
    plot_opacity = html.Div(sidebar_input('Opacity', 'number', f'{id}_opacity',
                                          min=0.10, max=1.0, value=0.85, step=0.01),
                            id=f'{id}_opacity_comp')
    buttons = dbc.Row([std_button('Reset', id=f'{id}_reset_button'),
                       std_button('Save', id=f'{id}_add_button')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})

    plot_controls = html.Div([#html.Div(sep, id=f'{id}_sep'),
                              num_bins, horizontal, density, bar_width, 
                              plot_color, plot_opacity,], id=f'{id}_plot_controls')
    univar_tabs = visual_tabs(id)
    visual_div = html.Div([univar_tabs, plot_controls, fig_controls(id)],
                          id=f'{id}_visual_div', style=hide)
    controls = html.Div([sep, plot_type, uni_column, visual_div, sep, buttons])

    figure = go.Figure(go.Bar(x=[], y=[]))

    summary_table = html.Div([], id=f'{id}_nums')

    code_string = '  '
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([code_box(code_string, id), error, sep,
                         summary_table,
                         dcc.Graph(figure=figure, id=f'{id}_graph', style=hide)])

    @callback(Output(f'{id}_graph', 'style'),
              Output(f'{id}_visual_div', 'style'),
              Output(f'{id}_horizontal_comp', 'style'),
              Output(f'{id}_density_comp', 'style'),
              Output(f'{id}_bar_width_comp', 'style'),
              Output(f'{id}_num_bins_comp', 'style'),
              Output(f'{id}_color_comp', 'style'),
              Output(f'{id}_opacity_comp', 'style'),
              #Output(f'{id}_config', 'style', allow_duplicate=True),
              # Output('debug_info', 'children'),
              Input(f'{id}_ptype', 'value'),
              prevent_initial_call=True)
    def update_control_styles(ptype):

        if ptype == 'numerical results':
            return hide, hide, hide, hide, hide, hide, hide, hide
        elif ptype == 'value counts':
            return show, show, show, show, show, hide, show, show
        elif ptype == 'hist':
            return show, show, hide, show, hide, show, show, show
        elif ptype == 'kde':
            return show, show, hide, hide, hide, hide, show, show
        elif ptype == 'boxplot':
            return show, show, show, hide, hide, hide, show, show
        else:
            return hide, hide, hide, hide, hide, hide, hide, hide

    @callback(Output(f'{id}_column', 'options'),
              Output(f'{id}_column', 'value'),
              Input(f'{id}_ptype', 'value'),
              State(f'{id}_column', 'value'),
              State('current', 'data'))
    def update_columns(ptype, column, current):

        idx = current['idx']
        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)

        all_labels = list(column_labels.index)

        is_number = input_data.apply(is_numeric_dtype, axis=0).values
        not_binary = ~input_data.apply(is_bool_dtype, axis=0).values
        num_labels = column_labels.index[is_number & not_binary]

        if ptype is None or ptype in ['numerical results', 'value counts']:
            if column is None:
                column = all_labels[0]
            return all_labels, column
        else:
            if column is None or column not in num_labels:
                column = num_labels[0]
            return num_labels, column

    @callback(Output(f'{id}_graph', 'figure'),
              Output(f'{id}_nums', 'children'),
              # Output(f'{id}_code', 'children'),
              Input(f'{id}_ptype', 'value'),
              Input(f'{id}_column', 'value'),
              Input(f'{id}_horizontal', 'value'),
              Input(f'{id}_density', 'value'),
              Input(f'{id}_bar_width', 'value'),
              Input(f'{id}_num_bins', 'value'),
              Input(f'{id}_color', 'value'),
              Input(f'{id}_opacity', 'value'),
              State(f'{id}_figure_width', 'value'),
              State(f'{id}_figure_height', 'value'),
              State(f'{id}_grid_switch', 'value'),
              State('current', 'data'))
    def update_figure_plot(ptype, column, horizontal, density, width, nbins, color, opacity,
                           fig_width, fig_height, grid, current):

        idx = current['idx']
        if idx is None or column is None:
            return go.Figure(go.Bar(x=[], y=[])), []

        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)
        col = input_data[column_labels[column]]
        show_density = len(density) > 0

        summary = []
        if ptype == 'numerical results':
            if is_numeric_dtype(col) and not is_bool_dtype(col):
                res = pd.DataFrame(col.describe()).T.round(4)
                res['count'] = res['count'].astype(int)
                res['missing'] = col.isnull().sum()
                res['dtype'] = col.dtype.__str__()
            else:
                counts = col.value_counts().sort_values()
                pr = col.value_counts(normalize=True).values
                res = pd.DataFrame({'count': [col.count()],
                                    'categories': [len(counts)],
                                    'Gini': [(pr * (1-pr)).sum().round(4)],
                                    'entropy': [-(pr * np.log2(pr)).sum().round(4)],
                                    'smallest': [f"{counts.index[0]}: {counts.values[0]}"],
                                    'largest': [f"{counts.index[-1]}: {counts.values[-1]}"],
                                    'missing': [col.isnull().sum()],
                                    'dtype': [col.dtype.__str__()]}, index=[column])
            figure =  go.Figure(go.Bar(x=[], y=[]))
            summary = data_frame(res, entire=False)
        elif ptype == 'value counts':
            res = col.value_counts(normalize=show_density)
            dist_label = 'Density' if show_density else 'Count'
            if len(horizontal) > 0:
                res = res.sort_values(ascending=True)
                xdata, ydata, orientation = res.values, res.index, 'h'
                ylabel, xlabel = column, dist_label
                #tickangle = 0
            else:
                ydata, xdata, orientation = res.values, res.index, 'v'
                xlabel, ylabel = column, dist_label

            barchart = go.Bar(x=xdata, y=ydata, width=width,
                              marker_color=color, opacity=opacity, orientation=orientation)
            layout = go.Layout(xaxis=dict(title=xlabel), yaxis=dict(title=ylabel))
            figure = go.Figure(barchart, layout=layout)
            summary = []
        elif ptype == 'hist':
            histnorm, ylabel = ('', 'Count') if not show_density else ('probability density', 'Density')
            layout = go.Layout(xaxis=dict(title=column), yaxis=dict(title=ylabel))
            hy, hx, ax = plt.hist(col, density=show_density, bins=nbins)
            xx = np.concatenate((hx[None, :-1], hx[None, 1:]), axis=0).T.flatten()
            yy = np.concatenate((hy[None, :], hy[None, :]), axis=0).T.flatten()
            h = str(color).lstrip('#')
            fill_rgba = ', '.join(tuple(str(int(h[i:i+2], 16)) for i in (0, 2, 4))) + f', {opacity}'
            histogram = go.Scatter(x=xx, y=yy, fill='tozeroy',
                                   line=dict(width=0), fillcolor=f'rgba({fill_rgba})')
            figure = go.Figure(histogram, layout=layout)
        elif ptype == 'kde':
            line = sns.kdeplot(col).lines[0]
            xx, yy = line.get_data()
            layout = go.Layout(xaxis=dict(title=column), yaxis=dict(title='Density'))
            h = str(color).lstrip('#')
            fill_rgba = ', '.join(tuple(str(int(h[i:i+2], 16)) for i in (0, 2, 4))) + f', {opacity-0.1}'
            line_rgba = ', '.join(tuple(str(int(h[i:i+2], 16)) for i in (0, 2, 4))) + f', {opacity}'
            scatter = go.Scatter(x=xx, y=yy, mode='lines',
                                 fill='tozeroy', fillcolor=f'rgba({fill_rgba})',
                                 line_color=f'rgba({line_rgba})', opacity=opacity, line_width=2)
            figure = go.Figure(scatter, layout=layout)
        elif ptype == 'boxplot':
            if len(horizontal) > 0:
                boxplot = go.Box(x=col, name='', orientation='h', whiskerwidth=0.5,
                                 fillcolor=color, line_color='black', opacity=opacity)
                layout = go.Layout(boxmode="group", boxgap=0.85, boxgroupgap=0,
                                   yaxis=dict(title=' '), xaxis=dict(title=column))
            else:
                boxplot = go.Box(y=col, name='', orientation='v', whiskerwidth=0.5,
                                 fillcolor=color, line_color='black', opacity=opacity)
                layout = go.Layout(boxmode="group", boxgap=0.85, boxgroupgap=0,
                                   xaxis=dict(title=' '), yaxis=dict(title=column))
            figure = go.Figure(boxplot, layout=layout)
        else:
            figure = go.Figure(go.Bar(x=[], y=[]))

        figure.update_layout(margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor='white',
                             width=fig_width, height=fig_height)
        figure.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        figure.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        if len(grid) > 0:
            figure.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
            figure.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
        else:
            figure.update_xaxes(showgrid=False)
            figure.update_yaxes(showgrid=False)

        return figure, summary

    @callback(Output(f'{id}_code', 'children'),
              Input(f'{id}_ptype', 'value'),
              Input(f'{id}_column', 'value'),
              Input(f'{id}_horizontal', 'value'),
              Input(f'{id}_density', 'value'),
              Input(f'{id}_bar_width', 'value'),
              Input(f'{id}_num_bins', 'value'),
              Input(f'{id}_color', 'value'),
              Input(f'{id}_opacity', 'value'),
              Input(f'{id}_figure_width', 'value'),
              Input(f'{id}_figure_height', 'value'),
              Input(f'{id}_grid_switch', 'value'),
              Input(f'{id}_xrotate', 'value'),
              State('current', 'data'))
    def update_code(ptype, column, horizontal, density, width, nbins, color, opacity,
                    fig_width, fig_height, grid, tickangle, current):

        idx = current['idx']
        if idx is None or column is None:
            return ''
        input_name, input_data = nodes[idx].content
        if column is None:
            return f'{input_name}'
        column_labels = index_labels(input_data.columns)
        col = input_data[column_labels[column]]

        code = univariate_code(input_name, col,
                               ptype, column, horizontal, density, width, nbins, color, opacity,
                               fig_width, fig_height, grid, tickangle)
        return code

    @callback(Output(f'{id}_graph', 'figure', allow_duplicate=True),
              Input(f'{id}_grid_switch', 'value'),
              Input(f'{id}_xrotate', 'value'),
              Input(f'{id}_figure_width', 'value'),
              Input(f'{id}_figure_height', 'value'),
              State(f'{id}_graph', 'figure'),
              prevent_initial_call=True)
    def update_figure_setting(switch, tickangle, width, height, figure):

        figure = go.Figure(figure)
        figure.update_layout(width=width, height=height)
        if len(switch) > 0:
            figure.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
            figure.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
        else:
            figure.update_xaxes(showgrid=False)
            figure.update_yaxes(showgrid=False)

        figure.update_xaxes(tickangle=-tickangle)

        return figure

    @callback(Output(f'{id}_ptype', 'value', allow_duplicate=True),
              Output(f'{id}_column', 'value', allow_duplicate=True),
              Input(f'{id}_reset_button', 'n_clicks'),
              State(f'{id}_ptype', 'options'),
              State(f'{id}_column', 'options'),
              prevent_initial_call=True)
    def update_univariate_reset(n_clicks, all_ptypes, all_columns):

        return all_ptypes[0], all_columns[0]

    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_add_button', 'n_clicks'),
              State('blueprint', 'figure'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_ptype', 'value'),
              State(f'{id}_graph', 'figure'),
              State(f'{id}_nums', 'children'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_univariate_add(n_clicks, figure, tabs, current, ptype, visual_figure, nums, code_markdown):

        code = code_markdown.replace('```python\n', '').replace('\n```', '')

        idx = current['idx']
        input_name = nodes[idx].content[0]
        if ptype == 'numerical results':
            visual_data = nums
        else:
            visual_data = visual_figure
        content = input_name, visual_data, 'Univariate distribution'

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'visual')

    return controls, previews


def bar_visual(current, nodes, id):

    global output_data

    idx = current['idx']
    input_name, input_data = nodes[idx].content

    all_columns = index_labels(input_data.columns)
    column_labels = list(all_columns.index)
    is_number = input_data.apply(is_numeric_dtype, axis=0).values
    not_binary = ~input_data.apply(is_bool_dtype, axis=0).values
    num_labels = list(all_columns.index[is_number & not_binary])
    bar_list = [{'column': None, 'color': None}]

    ydata = sidebar_dropdown('Bar data', num_labels, None, id=f'{id}_ydata')
    color = sidebar_input('Bar color', 'color', f'{id}_color', value='#1f77b4')
    add = dbc.Row([std_button('Add Bar', id=f'{id}_add', width=120)],justify='end',
                  style={'margin-left': 3, 'margin-right': 3, 'margin-top': 0, 'margin-bottom': 5})
    usex = sidebar_switch('Label data', 'no/yes', f'{id}_usex')
    xdata = html.Div(sidebar_dropdown('', column_labels, None, id=f'{id}_xdata'), id=f'{id}_xdata_comp', style=hide)

    btype = sidebar_dropdown('Type of bars', ['clustered', 'stacked'],
                             'clustered', id=f'{id}_btype')
    horizontal = sidebar_switch('Horizontal', 'no/yes', f'{id}_horizontal')
    bar_width = sidebar_input('Bar width', 'number', f'{id}_bar_width',
                              min=0.1, max=1.0, value=0.75, step=0.05)
    plot_opacity = html.Div(sidebar_input('Opacity', 'number', f'{id}_opacity',
                                          min=0.10, max=1.0, value=0.85, step=0.01),
                            id=f'{id}_opacity_comp')

    store = dcc.Store('bar_list', data=bar_list)
    buttons = dbc.Row([std_button('Reset', id=f'{id}_reset_button'),
                       std_button('Save', id=f'{id}_add_button')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    
    plot_controls = html.Div([ydata, color, add, sep, usex, xdata, sep,
                              plot_opacity, bar_width, horizontal, btype], id=f'{id}_plot_controls')
    bar_tabs = visual_tabs(id)    
    controls = html.Div([store, sep, bar_tabs, plot_controls, 
                         fig_controls(id, more=True), sep, buttons])

    figure = go.Figure(go.Bar(x=[], y=[]))
    code_string = '  '
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([code_box(code_string, id), error, sep,
                         dcc.Graph(figure=figure, id=f'{id}_graph')])

    @callback(Output(f'{id}_xdata_comp', 'style'),
              Output(f'{id}_xdata', 'value'),
              Input(f'{id}_usex', 'value'))
    def update_xdata_show(usex):

        if len(usex) > 0:
            return show, None
        else:
            return hide, None

    @callback(Output(f'{id}_graph', 'style'),
              Output(f'{id}_add', 'disabled'),
              Output(f'{id}_add_button', 'disabled'),
              Input('bar_list', 'data'))
    def update_bar_disabled(blist):

        no_bar = True
        for b in blist:
            if b['column'] is not None:
                no_bar = False
                break
        if no_bar:
            return hide, True, True
        else:
            return show, False, False

    @callback(Output(f'{id}_graph', 'figure'),
              Output('bar_list', 'data'),
              Input(f'{id}_ydata', 'value'),
              Input(f'{id}_color', 'value'),
              Input(f'{id}_xdata', 'value'),
              Input(f'{id}_bar_width', 'value'),
              Input(f'{id}_opacity', 'value'),
              Input(f'{id}_horizontal', 'value'),
              Input(f'{id}_btype', 'value'),
              State(f'{id}_legpos', 'value'),
              State(f'{id}_xlabel', 'value'),
              State(f'{id}_ylabel', 'value'),
              State(f'{id}_figure_width', 'value'),
              State(f'{id}_figure_height', 'value'),
              State(f'{id}_grid_switch', 'value'),
              State('current', 'data'),
              State('bar_list', 'data'))
    def update_barchart(ydata, color, xdata, bwidth, opacity, horizontal, btype,
                        legpos, xlabel, ylabel, fig_width, fig_height, grid, current, blist):

        idx = current['idx']
        if idx is None:
            return go.Figure(go.Bar(x=[], y=[])), blist

        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)

        if xdata is None:
            xvalues = input_data.index
            if isinstance(xvalues, pd.MultiIndex):
                xvalues = pd.Series([f"({','.join(np.array(i).astype(str))})"
                                     for i in xvalues], name=xvalues.name)
        else:
            xvalues = input_data[column_labels[xdata]]
        #yvalues = input_data[column_labels[ydata]]

        if ydata is not None:
            current_bar = blist[-1]
            current_bar['column'] = column_labels[ydata]
            current_bar['color'] = color

        columns = []
        colors = []
        for b in blist:
            if b['column'] is None:
                continue
            columns.append(b['column'])
            colors.append(b['color'])
        if btype == 'stacked':
            barmode = 'stack'
            num_bar = 1
        else:
            barmode = 'group'
            num_bar = len(columns)
        #yvalues = input_data[column_labels[columns]]
        bar_data = []
        for col, color in zip(columns, colors):
            name = col
            if isinstance(col, Iterable) and not isinstance(col, str):
                name = f"({', '.join(col)})"
                col = tuple(col)
            yvalues = input_data[col]
            if len(horizontal) > 0:
                bar_data.append(go.Bar(x=yvalues, y=xvalues, name=name, width=bwidth/num_bar,
                                       marker_color=color, opacity=opacity, orientation='h'))
            else:
                bar_data.append(go.Bar(x=xvalues, y=yvalues, name=name, width=bwidth/num_bar,
                                       marker_color=color, opacity=opacity, orientation='v'))

        figure = go.Figure(bar_data)
        figure.update_layout(bargap=1 - bwidth)
        ya, yy = ('top', 0.99) if 'upper' in legpos else ('bottom', 0.01)
        xa, xx = ('left', 0.01) if 'left' in legpos else ('right', 0.99)
        figure.update_layout(legend=dict(yanchor=ya, xanchor=xa, x=xx, y=yy,
                                         bordercolor='Black', borderwidth=1))
        figure.update_layout(margin=dict(l=10, r=10, t=40, b=10), barmode=barmode,
                             plot_bgcolor='white', width=fig_width, height=fig_height)
        figure.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        figure.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        figure.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        if len(grid) > 0:
            figure.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
            figure.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
        else:
            figure.update_xaxes(showgrid=False)
            figure.update_yaxes(showgrid=False)

        return figure, blist

    @callback(Output(f'{id}_ydata', 'value'),
              Output(f'{id}_color', 'value'),
              Output('bar_list', 'data', allow_duplicate=True),
              Input(f'{id}_add', 'n_clicks'),
              State('bar_list', 'data'),
              prevent_initial_call=True)
    def update_add_bar(n_clicks, bar_list):

        if n_clicks is None:
            n_clicks = 0

        all_colors = list(mpl.rcParams['axes.prop_cycle'])
        num_colors = len(all_colors)
        next_color = all_colors[n_clicks % num_colors]['color']
        bar_list.append({'column': None, 'color': None})

        return None, next_color, bar_list

    @callback(Output(f'{id}_graph', 'figure', allow_duplicate=True),
              Input(f'{id}_legpos', 'value'),
              Input(f'{id}_xlabel', 'value'),
              Input(f'{id}_ylabel', 'value'),
              Input(f'{id}_xrotate', 'value'),
              Input(f'{id}_grid_switch', 'value'),
              Input(f'{id}_figure_width', 'value'),
              Input(f'{id}_figure_height', 'value'),
              State(f'{id}_graph', 'figure'),
              prevent_initial_call=True)
    def update_figure_setting(legpos, xlabel, ylabel, tickangle, grid, width, height, figure):

        figure = go.Figure(figure)
        figure.update_layout(width=width, height=height)
        ya, yy = ('top', 0.99) if 'upper' in legpos else ('bottom', 0.01)
        xa, xx = ('left', 0.01) if 'left' in legpos else ('right', 0.99)
        figure.update_layout(legend=dict(yanchor=ya, xanchor=xa, x=xx, y=yy,
                                         bordercolor='Black', borderwidth=1))
        figure.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        if len(grid) > 0:
            figure.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
            figure.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
        else:
            figure.update_xaxes(showgrid=False)
            figure.update_yaxes(showgrid=False)

        figure.update_xaxes(tickangle=-tickangle)

        return figure

    @callback(Output(f'{id}_code', 'children'),
              Input('bar_list', 'data'),
              #Input(f'{id}_color', 'value'),
              Input(f'{id}_xdata', 'value'),
              Input(f'{id}_bar_width', 'value'),
              Input(f'{id}_opacity', 'value'),
              Input(f'{id}_horizontal', 'value'),
              Input(f'{id}_btype', 'value'),
              Input(f'{id}_legpos', 'value'),
              Input(f'{id}_xlabel', 'value'),
              Input(f'{id}_ylabel', 'value'),
              Input(f'{id}_figure_width', 'value'),
              Input(f'{id}_figure_height', 'value'),
              Input(f'{id}_grid_switch', 'value'),
              Input(f'{id}_xrotate', 'value'),
              State('current', 'data'))
    def update_code(blist, xdata, bwidth, opacity, horizontal, btype,
                    legpos, xlabel, ylabel, 
                    fig_width, fig_height, grid, tickangle, current):

        idx = current['idx']
        if idx is None:
            return ''
        input_name, input_data = nodes[idx].content

        code = barchart_code(input_name, input_data,
                             blist, xdata, bwidth, opacity, horizontal, btype,
                             legpos, xlabel, ylabel,
                             fig_width, fig_height, grid, tickangle)
        return code

    @callback(Output(f'{id}_ydata', 'value', allow_duplicate=True),
              Output(f'{id}_color', 'value', allow_duplicate=True),
              Output(f'{id}_add', 'disabled', allow_duplicate=True),
              Output(f'{id}_add', 'n_clicks', allow_duplicate=True),
              Output('bar_list', 'data', allow_duplicate=True),
              Input(f'{id}_reset_button', 'n_clicks'),
              prevent_initial_call=True)
    def update_bar_reset_button(n_clicks):

        return None, '#1f77b4', True, 0, [{'column': None, 'color': True}]

    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_add_button', 'n_clicks'),
              State('blueprint', 'figure'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_graph', 'figure'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_barchart_add_button(n_clicks, figure, tabs, current, visual_figure, code_markdown):

        code = code_markdown.replace('```python\n', '').replace('\n```', '')

        idx = current['idx']
        input_name = nodes[idx].content[0]
        content = input_name, visual_figure, 'Bar chart'

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'visual')

    return controls, previews


def scatter_visual(current, nodes, id):

    global output_data

    idx = current['idx']
    input_name, input_data = nodes[idx].content

    all_columns = index_labels(input_data.columns)
    column_labels = list(all_columns.index)
    is_number = input_data.apply(is_numeric_dtype, axis=0).values
    not_binary = ~input_data.apply(is_bool_dtype, axis=0).values
    num_labels = list(all_columns.index[is_number & not_binary])

    xdata = sidebar_dropdown('X-data', num_labels, None, id=f'{id}_xdata')
    ydata = sidebar_dropdown('Y-data', num_labels, None, id=f'{id}_ydata')
    size_switch = sidebar_switch('Variable size', 'no/yes', id=f'{id}_size_switch')
    size_scale = sidebar_slider('Size scale', -2, 2, 0.01, 0, id=f'{id}_size_scale')
    sdata = html.Div(sidebar_dropdown('Marker size data', num_labels, None, id=f'{id}_sdata'),
                     id=f'{id}_sdata_comp')
    color_switch = sidebar_switch('Variable color', 'no/yes', f'{id}_color_switch')
    color = html.Div(sidebar_input('Marker color', 'color', f'{id}_color', value='#1f77b4'),
                     id=f'{id}_color_comp')
    cdata = html.Div(sidebar_dropdown('Marker color data', column_labels, None, id=f'{id}_cdata'),
                     id=f'{id}_cdata_comp')
    plot_opacity = sidebar_input('Opacity', 'number', f'{id}_opacity',
                                 min=0.10, max=1.0, value=0.85, step=0.01)

    buttons = dbc.Row([std_button('Reset', id=f'{id}_reset_button'),
                       std_button('Save', id=f'{id}_add_button')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})

    scatter_tab = visual_tabs(id)
    plot_controls = html.Div([xdata, ydata, sep, 
                              size_switch, sdata, size_scale, sep,
                              color_switch, color, cdata, plot_opacity], id=f'{id}_plot_controls')
    controls = html.Div([sep, scatter_tab, plot_controls,
                         fig_controls(id, more=True), sep, buttons])

    figure = go.Figure(go.Bar(x=[], y=[]))
    code_string = '  '
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([code_box(code_string, id), error, sep,
                         dcc.Graph(figure=figure, id=f'{id}_graph')])
    
    @callback(Output(f'{id}_sdata_comp', 'style'),
              Output(f'{id}_size_scale', 'value'),
              Input(f'{id}_size_switch', 'value'))
    def update_size_switch(size_change):

        if len(size_change) > 0:
            return show, 0
        else:
            return hide, 0
    
    @callback(Output(f'{id}_color_comp', 'style'),
              Output(f'{id}_cdata_comp', 'style'),
              Input(f'{id}_color_switch', 'value'))
    def update_color_switch(color_change):

        if len(color_change) > 0:
            return hide, show
        else:
            return show, hide
    
    @callback(Output(f'{id}_graph', 'style'),
              Input(f'{id}_xdata', 'value'),
              Input(f'{id}_ydata', 'value'),)
    def update_graph_style(xdata, ydata):

        if not xdata or not ydata:
            return hide
        else:
            return show
        
    @callback(Output(f'{id}_graph', 'figure'),
              Input(f'{id}_xdata', 'value'),
              Input(f'{id}_ydata', 'value'),
              Input(f'{id}_size_switch', 'value'),
              Input(f'{id}_size_scale', 'value'),
              #Input(f'{id}_size', 'value'),
              Input(f'{id}_sdata', 'value'),
              Input(f'{id}_color_switch', 'value'),
              Input(f'{id}_color', 'value'),
              Input(f'{id}_cdata', 'value'),
              Input(f'{id}_opacity', 'value'),
              State(f'{id}_legpos', 'value'),
              State(f'{id}_xlabel', 'value'),
              State(f'{id}_ylabel', 'value'),
              State(f'{id}_figure_width', 'value'),
              State(f'{id}_figure_height', 'value'),
              State(f'{id}_grid_switch', 'value'),
              State('current', 'data'))
    def update_scatterplot(xdata, ydata,
                           size_switch, scale, sdata,
                           color_switch, color, cdata, opacity,
                           legpos, xlabel, ylabel, fig_width, fig_height, grid,
                           current):

        idx = current['idx']
        if idx is None:
            return go.Figure(go.Bar(x=[], y=[]))
        
        if not ydata or not xdata:
            scatters = go.Scatter(x=[], y=[])
        else:
            input_name, input_data = nodes[idx].content
            column_labels = index_labels(input_data.columns)
            xcol = input_data[column_labels[xdata]]
            ycol = input_data[column_labels[ydata]]
        
            marker = dict(color=color, size=10 * (5**(scale/2)), opacity=opacity,
                          line=dict(width=0))
            if len(size_switch) > 0 and sdata:
                marker['size'] = (input_data[column_labels[sdata]]**0.5) / 0.6 * (5**(scale/2))
                marker['size'].fillna(0, inplace=True)
            if len(color_switch) > 0 and cdata:
                color_column = input_data[column_labels[cdata]]
                if is_numeric_dtype(color_column) and not is_bool_dtype(color_column):
                    marker['color'] = color_column
                    marker['showscale'] = True
                    scatters = go.Scatter(x=xcol, y=ycol, mode='markers',
                                          marker=marker, marker_colorscale='Viridis')
                else:
                    scatters = []
                    cats = color_column.unique()
                    all_colors = list(mpl.rcParams['axes.prop_cycle'])
                    num_colors = len(all_colors)
                    #if isinstance(marker['size'], pd.Series):
                    #    marker_size = marker['size'].copy()
                    #else:
                    marker_size = marker['size']
                    for i, c in enumerate(cats):
                        idx = color_column == c
                        xs, ys = xcol.loc[idx], ycol.loc[idx]
                        next_color = all_colors[i % num_colors]['color']
                        marker['color'] = next_color
                        if isinstance(marker_size, Iterable):
                            marker['size'] = marker_size[idx]
                        scatters.append(go.Scatter(x=xs, y=ys, mode='markers',
                                                   marker=marker, name=str(c)))
            else:
                scatters = go.Scatter(x=xcol, y=ycol, mode='markers', marker=marker)

        figure = go.Figure(scatters)
        
        ya, yy = ('top', 0.99) if 'upper' in legpos else ('bottom', 0.01)
        xa, xx = ('left', 0.01) if 'left' in legpos else ('right', 0.99)
        figure.update_layout(legend=dict(yanchor=ya, xanchor=xa, x=xx, y=yy,
                                         bordercolor='Black', borderwidth=1))
        figure.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                             plot_bgcolor='white', width=fig_width, height=fig_height)
        figure.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        figure.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        figure.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        if len(grid) > 0:
            figure.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
            figure.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
        else:
            figure.update_xaxes(showgrid=False)
            figure.update_yaxes(showgrid=False)

        return figure

    @callback(Output(f'{id}_graph', 'figure', allow_duplicate=True),
              Input(f'{id}_legpos', 'value'),
              Input(f'{id}_xlabel', 'value'),
              Input(f'{id}_ylabel', 'value'),
              Input(f'{id}_xrotate', 'value'),
              Input(f'{id}_grid_switch', 'value'),
              Input(f'{id}_figure_width', 'value'),
              Input(f'{id}_figure_height', 'value'),
              State(f'{id}_graph', 'figure'),
              prevent_initial_call=True)
    def update_figure_setting(legpos, xlabel, ylabel, tickangle, grid, width, height, figure):

        figure = go.Figure(figure)
        figure.update_layout(width=width, height=height)
        ya, yy = ('top', 0.99) if 'upper' in legpos else ('bottom', 0.01)
        xa, xx = ('left', 0.01) if 'left' in legpos else ('right', 0.99)
        figure.update_layout(legend=dict(yanchor=ya, xanchor=xa, x=xx, y=yy,
                                         bordercolor='Black', borderwidth=1))
        figure.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        if len(grid) > 0:
            figure.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
            figure.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
        else:
            figure.update_xaxes(showgrid=False)
            figure.update_yaxes(showgrid=False)

        figure.update_xaxes(tickangle=-tickangle)

        return figure
    
    @callback(Output(f'{id}_code', 'children'),
              Input(f'{id}_xdata', 'value'),
              Input(f'{id}_ydata', 'value'),
              Input(f'{id}_size_switch', 'value'),
              Input(f'{id}_size_scale', 'value'),
              Input(f'{id}_sdata', 'value'),
              Input(f'{id}_color_switch', 'value'),
              Input(f'{id}_color', 'value'),
              Input(f'{id}_cdata', 'value'),
              Input(f'{id}_opacity', 'value'),
              Input(f'{id}_legpos', 'value'),
              Input(f'{id}_xlabel', 'value'),
              Input(f'{id}_ylabel', 'value'),
              Input(f'{id}_figure_width', 'value'),
              Input(f'{id}_figure_height', 'value'),
              Input(f'{id}_grid_switch', 'value'),
              Input(f'{id}_xrotate', 'value'),
              State('current', 'data'))
    def update_code(xdata, ydata, size_switch, scale, sdata,
                    color_switch, color, cdata, opacity,
                    legpos, xlabel, ylabel, 
                    fig_width, fig_height, grid, tickangle, current):

        idx = current['idx']
        if idx is None:
            return ''
        input_name, input_data = nodes[idx].content

        code = scatterplot_code(input_name, input_data, xdata, ydata,
                                size_switch, scale, sdata, color_switch, color, cdata, opacity,
                                legpos, xlabel, ylabel,
                                fig_width, fig_height, grid, tickangle)
        return code
    
    @callback(Output(f'{id}_xdata', 'value', allow_duplicate=True),
              Output(f'{id}_ydata', 'value', allow_duplicate=True),
              Output(f'{id}_size_switch', 'value', allow_duplicate=True),
              Output(f'{id}_size_scale', 'value', allow_duplicate=True),
              Output(f'{id}_sdata', 'value', allow_duplicate=True),
              Output(f'{id}_color_switch', 'value', allow_duplicate=True),
              Output(f'{id}_color', 'value', allow_duplicate=True),
              Output(f'{id}_cdata', 'value', allow_duplicate=True),
              Output(f'{id}_opacity', 'value', allow_duplicate=True),
              Input(f'{id}_reset_button', 'n_clicks'),
              prevent_initial_call=True)
    def update_scatterplot_reset_button(n_clicks):

        return None, None, [], 0, None, [], '#1f77b4', None, 0.85

    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_add_button', 'n_clicks'),
              State('blueprint', 'figure'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_graph', 'figure'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_scatterplot_add_button(n_clicks, figure, tabs, current, visual_figure, code_markdown):

        code = code_markdown.replace('```python\n', '').replace('\n```', '')

        idx = current['idx']
        input_name = nodes[idx].content[0]
        content = input_name, visual_figure, 'Scatterplot'

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'visual')
    
    return controls, previews


def lineplot_visual(current, nodes, id):

    global output_data

    idx = current['idx']
    input_name, input_data = nodes[idx].content

    all_columns = index_labels(input_data.columns)
    is_number = input_data.apply(is_numeric_dtype, axis=0).values
    not_binary = ~input_data.apply(is_bool_dtype, axis=0).values
    is_ts = pd.Series(input_data.dtypes).apply(lambda x: np.issubdtype(x, np.datetime64)).values
    num_labels = list(all_columns.index[is_number & not_binary])
    xcol_labels = list(all_columns.index[(is_number & not_binary) | is_ts])

    ydata = sidebar_dropdown('Y-data', num_labels, None, id=f'{id}_ydata')
    trans = sidebar_dropdown('Transform', ['change', 'fractional change', 'moving average'], None,
                             id=f'{id}_trans', placeholder='index')
    period = sidebar_input('Period', 'number', f'{id}_period', min=1, max=100, step=1, value=1)
    xdata = sidebar_dropdown('X-data', xcol_labels, None, id=f'{id}_xdata')
    linestyle = sidebar_inline_dropdown('Line', ['solid', 'dash', 'dot', 'dashdot'], 'solid', id=f'{id}_linestyle')
    linewidth = sidebar_input('Line width', 'number', id=f'{id}_linewidth', min=0.5, max=4, step=0.5, value=1.0)
    size_scale = sidebar_slider('Size scale', -1, 1, 0.01, 0, id=f'{id}_size_scale')
    color = sidebar_input('Color', 'color', f'{id}_color', value='#1f77b4')
    marker = sidebar_inline_dropdown('Marker', ['none', 'circle', 'square', 'dot', 'diamond', 'triangle'],
                                     'none', id=f'{id}_marker')
    add = dbc.Row([std_button('Add Line', id=f'{id}_add', width=120)],justify='end',
                  style={'margin-left': 3, 'margin-right': 3, 'margin-top': 0, 'margin-bottom': 5})

    line_list = [{'y': None, 'trans': None, 'period': 1, 'x': None,
                  'linewidth': None, 'linestyle': None, 'color': None,
                  'marker': None, 'scale': 0}]
    store = dcc.Store('line_list', data=line_list)
    buttons = dbc.Row([std_button('Reset', id=f'{id}_reset_button'),
                       std_button('Save', id=f'{id}_add_button')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})

    scatter_tab = visual_tabs(id)
    plot_controls = html.Div([ydata, trans, period, xdata, sep,
                              linestyle, linewidth, marker, size_scale, color, add],
                              id=f'{id}_plot_controls')
    controls = html.Div([store, sep, scatter_tab, plot_controls,
                         fig_controls(id, more=True), sep, buttons])

    figure = go.Figure(go.Bar(x=[], y=[]))
    code_string = '  '
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([code_box(code_string, id), error, sep,
                         dcc.Graph(figure=figure, id=f'{id}_graph')])
    
    @callback(Output(f'{id}_graph', 'style'),
              Output(f'{id}_add', 'disabled'),
              Output(f'{id}_add_button', 'disabled'),
              Input(f'{id}_ydata', 'value'),
              Input('line_list', 'data'))
    def update_graph_style(ydata, llist):

        no_line = True
        for lin in llist:
            if lin['y']:
                no_line = False
                break

        if no_line:
            return hide, True, True
        else:
            disable_add = True if not ydata else False 
            return show, disable_add, False
        
    @callback(Output(f'{id}_period_div', 'style'),
              Input(f'{id}_trans', 'value'))
    def update_period_style(trans):

        if not trans:
            return hide
        else:
            return show
        
    @callback(Output(f'{id}_size_scale_div', 'style'),
              Input(f'{id}_marker', 'value'))
    def update_scale_style(marker):

        if not marker or marker == 'none':
            return hide
        else:
            return show
    
    @callback(Output(f'{id}_graph', 'figure'),
              Output('line_list', 'data'),
              Input(f'{id}_ydata', 'value'),
              Input(f'{id}_trans', 'value'),
              Input(f'{id}_period', 'value'),
              Input(f'{id}_xdata', 'value'),
              Input(f'{id}_linestyle', 'value'),
              Input(f'{id}_linewidth', 'value'),
              Input(f'{id}_marker', 'value'),
              Input(f'{id}_size_scale', 'value'),
              Input(f'{id}_color', 'value'),
              State(f'{id}_legpos', 'value'),
              State(f'{id}_xlabel', 'value'),
              State(f'{id}_ylabel', 'value'),
              State(f'{id}_figure_width', 'value'),
              State(f'{id}_figure_height', 'value'),
              State(f'{id}_grid_switch', 'value'),
              State(f'{id}_xrotate', 'value'),
              State('line_list', 'data'),
              State('current', 'data'))
    def update_lineplot_figure(ydata, trans, period, xdata, 
                               linestyle, linewidth, marker, scale, color,
                               legpos, xlabel, ylabel, fig_width, fig_height, grid, xrotate,
                               llist, current):

        idx = current['idx']
        if idx is None:
            return go.Figure(go.Bar(x=[], y=[])), llist
        
        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)

        if ydata is not None:
            current_line = llist[-1]
            current_line['y'] = column_labels[ydata]
            current_line['trans'] = trans
            current_line['period'] = period
            if xdata:
                current_line['x'] = column_labels[xdata]
            else:
                current_line['x'] = None
            current_line['linestyle'] = linestyle
            current_line['linewidth'] = linewidth
            current_line['marker'] = marker
            current_line['color'] = color
            current_line['scale'] = scale

        lines = [go.Scatter(x=[], y=[], mode='lines')]
        for lin in llist:
            if lin['y'] is None:
                continue
            if isinstance(lin['y'], Iterable) and not isinstance(lin['y'], str):
                lin['y'] = tuple(lin['y'])
                data_label = '_'.join(pd.Series(lin['y']).astype(str))
            else:
                data_label = f"{lin['y']}"
            ycol = input_data[lin['y']]
            if lin['trans'] == 'change':
                ycol = ycol.diff(lin['period'])
                data_label = f"{lin['period']}-period change of {data_label}"
            elif lin['trans'] == 'fractional change':
                ycol = ycol.pct_change(lin['period'])
                data_label = f"{lin['period']}-period fractional change of {data_label}"
            elif lin['trans'] == 'moving average':
                ycol = ycol.rolling(lin['period']).mean()
                data_label = f"{lin['period']}-period moving average of {data_label}"
            if not lin['x'] or lin['x'] == '':
                xcol = input_data.index
            else:
                if isinstance(lin['x'], Iterable) and not isinstance(lin['y'], str):
                    lin['x'] = tuple(lin['x'])
                xcol = input_data[lin['x']]
            
            if not lin['marker'] or lin['marker'] == 'none':
                mode = 'lines'
                marker_dict = {}
            else:
                mode = 'lines+markers'
                symbols = {'circle': 'circle',
                           'dot': 'circle',
                           'square': 'square',
                           'diamond': 'diamond',
                           'triangle': 'triangle-up'}
                symbol = symbols[lin['marker']]
                markersize = 4 if lin['marker'] == 'dot' else 10
                marker_dict = dict(size=markersize*(5**(lin['scale']/2)),
                                   symbol=symbol, color=lin['color'])
            
            line_dict = dict(color=lin['color'], width=lin['linewidth']*1.8, dash=lin['linestyle'])
            lines.append(go.Scatter(x=xcol, y=ycol, mode=mode,
                                    line=line_dict, marker=marker_dict, name=data_label))
        
        figure = go.Figure(lines)
        
        ya, yy = ('top', 0.99) if 'upper' in legpos else ('bottom', 0.01)
        xa, xx = ('left', 0.01) if 'left' in legpos else ('right', 0.99)
        figure.update_layout(legend=dict(yanchor=ya, xanchor=xa, x=xx, y=yy,
                                         bordercolor='Black', borderwidth=1))
        figure.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                             plot_bgcolor='white', width=fig_width, height=fig_height)
        figure.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        figure.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        figure.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        if len(grid) > 0:
            figure.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
            figure.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
        else:
            figure.update_xaxes(showgrid=False)
            figure.update_yaxes(showgrid=False)

        return figure, llist
    
    @callback(Output(f'{id}_ydata', 'value'),
              # Output(f'{id}_xdata', 'value'),
              Output(f'{id}_linestyle', 'value'),
              Output(f'{id}_linewidth', 'value'),
              Output(f'{id}_marker', 'value'),
              Output(f'{id}_color', 'value'),
              Output('line_list', 'data', allow_duplicate=True),
              Input(f'{id}_add', 'n_clicks'),
              State('line_list', 'data'),
              prevent_initial_call=True)
    def update_add_line(n_clicks, line_list):

        if n_clicks is None:
            n_clicks = 0

        all_colors = list(mpl.rcParams['axes.prop_cycle'])
        num_colors = len(all_colors)
        next_color = all_colors[n_clicks % num_colors]['color']
        
        line_list.append({'y': None, 'trans': None, 'period': 1, 'x': None,
                          'linewidth': None, 'linestyle': None, 'color': None,
                          'marker': None, 'scale': 0})

        return None, 'solid', 1.0, 'none', next_color, line_list

    @callback(Output(f'{id}_graph', 'figure', allow_duplicate=True),
              Input(f'{id}_legpos', 'value'),
              Input(f'{id}_xlabel', 'value'),
              Input(f'{id}_ylabel', 'value'),
              Input(f'{id}_xrotate', 'value'),
              Input(f'{id}_grid_switch', 'value'),
              Input(f'{id}_figure_width', 'value'),
              Input(f'{id}_figure_height', 'value'),
              State(f'{id}_graph', 'figure'),
              prevent_initial_call=True)
    def update_figure_setting(legpos, xlabel, ylabel, tickangle, grid, width, height, figure):

        figure = go.Figure(figure)
        figure.update_layout(width=width, height=height)
        ya, yy = ('top', 0.99) if 'upper' in legpos else ('bottom', 0.01)
        xa, xx = ('left', 0.01) if 'left' in legpos else ('right', 0.99)
        figure.update_layout(legend=dict(yanchor=ya, xanchor=xa, x=xx, y=yy,
                                         bordercolor='Black', borderwidth=1))
        figure.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        if len(grid) > 0:
            figure.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
            figure.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
        else:
            figure.update_xaxes(showgrid=False)
            figure.update_yaxes(showgrid=False)

        figure.update_xaxes(tickangle=-tickangle)

        return figure
    
    @callback(Output(f'{id}_code', 'children'),
              Input(f'{id}_legpos', 'value'),
              Input(f'{id}_xlabel', 'value'),
              Input(f'{id}_ylabel', 'value'),
              Input(f'{id}_figure_width', 'value'),
              Input(f'{id}_figure_height', 'value'),
              Input(f'{id}_grid_switch', 'value'),
              Input(f'{id}_xrotate', 'value'),
              Input('line_list', 'data'),
              State('current', 'data'))
    def update_lineplot_code(legpos, xlabel, ylabel, fig_width, fig_height, grid, xrotate, 
                             llist, current):
        
        idx = current['idx']
        if idx is None:
            return ''
        input_name, input_data = nodes[idx].content

        code = lineplot_code(input_name, llist,
                             legpos, xlabel, ylabel, fig_width, fig_height, grid, xrotate)
        
        return code
    
    @callback(Output(f'{id}_xdata', 'value', allow_duplicate=True),
              Output(f'{id}_ydata', 'value', allow_duplicate=True),
              Output(f'{id}_linestyle', 'value', allow_duplicate=True),
              Output(f'{id}_linewidth', 'value', allow_duplicate=True),
              Output(f'{id}_marker', 'value', allow_duplicate=True),
              Output(f'{id}_color', 'value', allow_duplicate=True),
              Output('line_list', 'data', allow_duplicate=True),
              Output(f'{id}_add', 'n_clicks'),
              Input(f'{id}_reset_button', 'n_clicks'),
              prevent_initial_call=True)
    def update_line_reset_button(n_clicks):

        line = {'y': None, 'x': None, 'linewidth': None, 'linestyle': None,
                'color': None, 'marker': None}

        return None, None, 'solid', 1.0, 'none', '#1f77b4', [line], None
    
    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_add_button', 'n_clicks'),
              State('blueprint', 'figure'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_graph', 'figure'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_line_add_button(n_clicks, figure, tabs, current, visual_figure, code_markdown):

        code = code_markdown.replace('```python\n', '').replace('\n```', '')

        idx = current['idx']
        input_name = nodes[idx].content[0]
        content = input_name, visual_figure, 'Line plot'

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'visual')
    
    return controls, previews
    
    return controls, previews


########### Model Pipeline ####################
def model_variables(current, nodes, steps, id, mtype='P'):

    idx = current['idx']
    input_name, input_data = nodes[idx].content
    if mtype == 'P':
        xlabel = 'Predictor variables'
        ylabel  ='Predicted variable'
    else:
        xlabel = 'Independent variable'
        ylabel = 'Dependent variable'

    all_columns = index_labels(input_data.columns)
    column_labels = list(all_columns.index)
    if mtype == 'P':
        predicted_options = column_labels
    else:
        num_columns = input_data.select_dtypes(include='number').columns
        predicted_options = list(index_labels(num_columns).index)

    predicted = sidebar_dropdown(ylabel, predicted_options, None, id=f'{id}_predicted')
    predictors = sidebar_dropdown(xlabel, column_labels, None,
                                  id=f'{id}_predictors', multi=True)
    
    buttons = dbc.Row([std_button('Back', id=f'{id}_back'),
                       std_button('Next', id=f'{id}_next')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    
    store = dcc.Store(f'{id}_pred_type', data='R')
    controls = html.Div([store,
                         dcc.Markdown('#### Model Pipeline'), sep,
                         dcc.Markdown('**Step 1: variable selection**'),
                         predicted, predictors, 
                         sep, buttons])
    
    table = html.Div(data_frame(input_data, entire=False), id=f'{id}_table')
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    row, col = input_data.shape
    ws = '&nbsp;'
    notes = f'<ul>\n<li><span style="background-color:rgba(248, 154, 54, 0.2)">{ws*5}</span>{ws*2} Dependent/predicted variable</li>\n'
    notes += f'<li><span style="background-color:rgba(49, 154, 206, 0.2)">{ws*5}</span>{ws*2} Independent/predictor variables</li>\n</ul>'
    code = code_box('', id) #if mtype == 'P' else html.Div(' ', f'{id}_code')
    previews = html.Div([dcc.Markdown('#### Preview'), sep,
                         code, error, sep,
                         dcc.Markdown(f'dataset: ```{input_name}```'),
                         dcc.Markdown(f'{row} rows $\\times$ {col} columns\n', mathjax=True),
                         table, 
                         dash_dangerously_set_inner_html.DangerouslySetInnerHTML(notes)])
    
    @callback(Output(f'{id}_predictors', 'options'),
              Output(f'{id}_predictors', 'value'),
              Output(f'{id}_pred_type', 'data'),
              Input(f'{id}_predicted', 'value'),
              State('current', 'data'))
    def update_predictor_options(predicted, current):

        idx = current['idx']
        input_data = nodes[idx].content[1]

        all_columns = index_labels(input_data.columns)

        pred_type = 'R'
        if predicted:
            col = input_data[all_columns[predicted]]
            if not is_numeric_dtype(col) or is_bool_dtype(col):
                pred_type = 'C'

        if predicted is None:
            return list(all_columns.index), None, pred_type
        else:
            options = list(all_columns.index.drop(predicted))
            return options, options, pred_type
    
    @callback(Output(f'{id}_code', 'children'),
              Output(f'{id}_next', 'disabled'),
              Input(f'{id}_predicted', 'value'),
              Input(f'{id}_predictors', 'value'),
              State('model_type', 'value'),
              State('current', 'data'))
    def update_next_disabled(predicted, predictors, mtype, current):

        idx = current['idx']
        input_name = nodes[idx].content[0]
        code = model_vars_code(input_name, predicted, predictors, mtype)
        if not predicted or not predictors:
            return code, True
        else:
            return code, False
    
    @callback(Output(f'{id}_table', 'children'),
              Input(f'{id}_predicted', 'value'),
              Input(f'{id}_predictors', 'value'),
              State('current', 'data'))
    def update_table(predicted, predictors, current):

        idx = current['idx']
        input_data = nodes[idx].content[1]
        
        extra_style = []
        if predicted is not None:
            extra_style += [{'if': {'column_id': '_'.join(predicted)},
                             'backgroundColor': 'rgba(248, 154, 54, 0.2)'}]
        if predictors is not None:
            cols = ['_'.join(c) for c in predictors]
            extra_style += [{'if': {'column_id': cols},
                             'backgroundColor': 'rgba(49, 154, 206, 0.2)'}]

        return data_frame(input_data, entire=False, extra_style=extra_style)

    @callback(Output('pred_vars_div', 'style', allow_duplicate=True),
              Output('exp_vars_div', 'style', allow_duplicate=True),
              Output('model_start_div', 'style', allow_duplicate=True),
              Output('model_type', 'value', allow_duplicate=True),
              Output('clear_model', 'data', allow_duplicate=True),
              Input(f'{id}_back', 'n_clicks'),
              State('model_type', 'value'),
              #State('model_page_idx', 'data'),
              prevent_initial_call=True)
    def update_back_button(n_clicks, mtype):

        if n_clicks is None:
            if mtype == 'Predictive modeling':
                return show, hide, hide, mtype, True
            elif mtype == 'Explanatory modeling':
                return hide, show, hide, mtype, True
            else:
                return hide, hide, show, mtype, True
        else:
            return hide, hide, show, None, True
    
    @callback(Output('pred_cats_div', 'children', allow_duplicate=True),
              Output('pred_vars_div', 'style', allow_duplicate=True),
              Output('pred_cats_div', 'style', allow_duplicate=True),
              Output('exp_fit_div', 'children', allow_duplicate=True),
              Output('exp_vars_div', 'style', allow_duplicate=True),
              Output('exp_fit_div', 'style', allow_duplicate=True),
              #Output('model_at_home', 'data', allow_duplicate=True),
              Output('clear_model', 'data', allow_duplicate=True),
              Input(f'{id}_next', 'n_clicks'),
              State('exp_fit_div', 'children'),
              State(f'{id}_predicted', 'value'),
              State(f'{id}_predictors', 'value'),
              State('model_type', 'value'),
              State('current', 'data'),
              #State('model_page_idx', 'data'),
              prevent_initial_call=True)
    def update_next_button(n_clicks, page, predicted, predictors, mtype, current):

        if n_clicks is None:
            if mtype == 'Predictive modeling':
                return [], show, hide, [], hide, hide, False
            elif mtype == 'Explanatory modeling':    
                return [], hide, hide, [], show, hide, False
        else:
            if mtype == 'Predictive modeling':
                controls, previews = pred_cats(current, nodes, steps, id='pred_cats',
                                               predicted=predicted, predictors=predictors)
                page = two_columns(controls, previews, id='pred_cats_page')
                return page, hide, show, [], hide, hide, False
            elif mtype == 'Explanatory modeling':
                controls, previews = exp_model_fit(current, nodes, id='exp_fit',
                                                   predicted=predicted, predictors=predictors)
                page = two_columns(controls, previews, id='exp_fit_page')
                return [], hide, hide, page, hide, show, False
            else:
                return page, hide, hide, [], hide, hide, False

    return controls, previews


def exp_model_fit(current, nodes, id, predicted='', predictors=[]):

    formula_value = f"{predicted} ~ {' + '.join(predictors)}"
    formula = sidebar_expr('Formula', id=f'{id}_formula', value=formula_value)
    fit = dbc.Row([std_button('Fit model', id=f'{id}_fit', width=120)], justify='end',
                  style={'margin-left': 3, 'margin-right': 3, 'margin-top': 0, 'margin-bottom': 5})
    ftype = sidebar_dropdown('Fitting function', ['ols', 'logit'], 'ols', id=f'{id}_ftype')
    buttons = dbc.Row([std_button('Back', id=f'{id}_back'),
                       std_button('Save', id=f'{id}_save')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})

    #store = dcc.Store('fit_result', data={})
    controls = html.Div([dcc.Markdown('#### Model Pipeline'), sep,
                         dcc.Markdown('**Step 2: model fitting**'),
                         formula, ftype, fit, sep, buttons])
    
    code_string = '  '
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([dcc.Markdown('#### Preview'), sep,
                         code_box(code_string, id), error, sep,
                         dcc.Markdown('**Results**'),
                         dcc.Markdown('', id=f'{id}_results')])
    
    @callback(Output(f'{id}_code', 'children'),
              Input(f'{id}_formula', 'value'),
              Input(f'{id}_ftype', 'value'),
              State('current', 'data'))
    def update_fit_code(formula, ftype, current):

        idx = current['idx']
        input_name = nodes[idx].content[0]

        return exp_fit_code(formula, input_name, ftype)
    
    @callback(Output(f'{id}_results', 'children'),
              Output(f'{id}_error', 'children'),
              Output(f'{id}_save', 'disabled'),
              # Output('fit_result', 'data'),
              Input(f'{id}_fit', 'n_clicks'),
              State(f'{id}_formula', 'value'),
              State(f'{id}_ftype', 'value'),
              State('current', 'data'))
    def update_fit_button(n_clicks, formula, ftype, current):

        if n_clicks is None:
            return '', '', True
        else:
            if ftype is not None:
                idx = current['idx']
                input_name, input_data = nodes[idx].content
                func = smf.ols if ftype == 'ols' else smf.logit
                try:
                    results = func(formula=formula, data=input_data).fit()
                    summary = results.summary().__str__()
                    return f'```\n{summary}\n```', '', False
                except Exception as err:
                    return '```\nNone\n```', f'\n\n**`Error`**`: {err}`', True
            else:
                return '', '', True

    @callback(Output('exp_fit_div', 'style', allow_duplicate=True),
              Output('exp_vars_div', 'style', allow_duplicate=True),
              Input(f'{id}_back', 'n_clicks'),
              prevent_initial_call=True)
    def update_back_button(n_clicks):

        if n_clicks is None:
            return show, hide
        else:
            return hide, show

    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_save', 'n_clicks'),
              State('blueprint', 'figure'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'{id}_ftype', 'value'),
              State(f'{id}_results', 'children'),
              State(f'{id}_code', 'children'),
              #State('blueprint', 'selectedData'),
              prevent_initial_call=True)
    def update_exp_save_button(n_clicks, figure, tabs, current,
                               ftype, summary, code_markdown):
        
        code = code_markdown.replace('```python\n', '').replace('\n```', '')

        idx = current['idx']
        input_name = nodes[idx].content[0]

        summary = summary.replace('```\n', '').replace('\n```', '')
        fit = 'logistic regression' if ftype == 'logit' else 'linear regression'
        model_info = f'Explanatory {fit} model fit on <b>{input_name}</b>'
        content = input_name, summary, model_info

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'model')
    
    return controls, previews


def exp_pipeline(current, nodes):

    mv_controls, mv_previews = model_variables(current, nodes, [], id='exp_vars', mtype='E')
    mv_step = html.Div(two_columns(mv_controls, mv_previews, id='exp_vars_page'),
                       id='exp_vars_div', style=hide)

    mf_controls, mf_previews = exp_model_fit(current, nodes,
                                             id='exp_fit', predicted=[], predictors=[])
    mf_step = html.Div(two_columns(mf_controls, mf_previews, id='exp_fit_page'),
                       id='exp_fit_div', style=hide)

    return [mv_step, mf_step]


def pred_cats(current, nodes, steps, id, predicted='', predictors=[]):

    idx = current['idx']
    input_name, input_data = nodes[idx].content
    column_labels = index_labels(input_data.columns)
    if predicted != '' and predictors:
        xcol = input_data[column_labels[predictors]]
        ycol = input_data[column_labels[predicted]]
        cond = is_numeric_dtype(ycol) and not is_bool_dtype(ycol)
        ftype = 'regression' if cond else 'classification'
        is_number = xcol.apply(is_numeric_dtype, axis=0).values
        is_bool = xcol.apply(is_bool_dtype, axis=0).values
        num_xcol = np.array(predictors)[is_number & (~is_bool)]
        cat_xcol = list(np.array(predictors)[~is_number])
    else:
        num_xcol = []
        cat_xcol = []
    extra_cats = sidebar_dropdown('Dummies for numerics', num_xcol, None,
                                  id=f'{id}_extra_cats', multi=True)
    buttons = dbc.Row([std_button('Back', id=f'{id}_back'),
                       std_button('Next', id=f'{id}_next')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    
    store = dcc.Store(id=f'{id}_xcat', data=cat_xcol)
    controls = html.Div([store,
                         dcc.Markdown('#### Model Pipeline'), sep,
                         dcc.Markdown('**Step 2: one hot encoder**'),
                         extra_cats, sep, buttons])
    
    code_string = '  '
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([dcc.Markdown('#### Preview'), sep,
                         code_box(code_string, id), error, sep,
                         html.Div([], id=f'{id}_diagram')])
    
    @callback(Output(f'{id}_diagram', 'children'),
              Output(f'{id}_code', 'children'),
              Input(f'{id}_extra_cats', 'value'),
              State(f'{id}_xcat', 'data'),
              State('current', 'data'))
    def update_dummy_diagram(extra_cats, xcat, current):

        idx = current['idx']
        input_name, input_data = nodes[idx].content
        column_labels = index_labels(input_data.columns)
        
        all_cats = xcat 
        if extra_cats is not None:
            all_cats += list(column_labels[extra_cats])

        if len(all_cats) > 0:
            ohe = OneHotEncoder(drop='first', sparse_output=False)
            to_dummies = ColumnTransformer(transformers=[('cats', ohe, all_cats)],
                                           remainder='passthrough')
            diagram = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(to_dummies._repr_html_())
            steps['dummy'] = to_dummies
        else:
            diagram = ['No categorical variable selected.\n']
        code = pred_cats_code(all_cats)

        return diagram, code

    @callback(Output('pred_cats_div', 'style', allow_duplicate=True),
              Output('pred_vars_div', 'style', allow_duplicate=True),
              Input(f'{id}_back', 'n_clicks'),
              prevent_initial_call=True)
    def update_back_button(n_clicks):

        if 'dummy' in steps:
            steps.pop('dummy')
        if n_clicks is None:
            return show, hide
        else:
            return hide, show
    
    @callback(Output('pred_reg_model_div', 'children', allow_duplicate=True),
              Output('pred_cats_div', 'style', allow_duplicate=True),
              Output('pred_reg_model_div', 'style', allow_duplicate=True),
              Input(f'{id}_next', 'n_clicks'),
              #State('pred_vars_predicted', 'value'),
              #State('current', 'data'),
              State('pred_vars_pred_type', 'data'),
              prevent_initial_call=True)
    def update_next_button(n_clicks, pred_type):

        if n_clicks is None:
            return [], show, hide
        else:
            #if is_numeric_dtype(col) and not is_bool_dtype(col):
            controls, previews = pred_regression_model(nodes, steps, pred_type, id='pred_reg_model')
            page = two_columns(controls, previews, id='pred_reg_model_page')
            #else:
            #    controls, previews = pred_classification_model(nodes, steps, id='pred_cls_model')
            #    page = two_columns(controls, previews, id='pred_cls_model_page')

            return page, hide, show

    return controls, previews


def pred_regression_model(nodes, steps, pred_type, id):

    scaling = sidebar_dropdown('Scaling of variables', ['StandardScaler', 'Normalizer'], None,
                               placeholder='none', id=f'{id}_scaling')
    redim = sidebar_switch('PCA', 'no/yes', id=f'{id}_pca_switch')
    num_comp = sidebar_expr('Number of components', id=f'{id}_nc', placeholder='None')

    if pred_type == 'R':
        model_options = ['LinearRegression', 'Ridge', 'Lasso',
                         'DecisionTreeRegressor', 'RandomForestRegressor']
    else:
        model_options = ['LogisticRegression',
                         'DecisionTreeClassifer', 'RandomForestClassifier']
    model_value = model_options[0]
    model = sidebar_dropdown('Model specifications', model_options, model_value,
                             id=f'{id}_model', clearable=False)
    alpha = sidebar_expr('Shrinkage parameter alpha', id=f'{id}_alpha', placeholder='1.0')
    cvalue = sidebar_expr('Shrinkage parameter C', id=f'{id}_cvalue', placeholder='1.0')
    max_depth = sidebar_expr('Max tree depth', id=f'{id}_max_depth', placeholder='None')
    min_sample_split = sidebar_expr('Min sample split',
                                    id=f'{id}_min_sample_split', placeholder='2')
    min_samples_leaf = sidebar_expr('Min samples leaf',
                                    id=f'{id}_min_samples_leaf', placeholder='1')
    max_features = sidebar_expr('Max number of features',
                                id=f'{id}_max_features', placeholder='None')
    max_leaf_nodes = sidebar_expr('Max number of leaf nodes',
                                  id=f'{id}_max_leaf_nodes', placeholder='None')
    
    buttons = dbc.Row([std_button('Back', id=f'{id}_back'),
                       std_button('Next', id=f'{id}_next')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    
    store = dcc.Store('reg_params')
    pred_type_string = 'regression' if pred_type == 'R' else 'classification'
    controls = html.Div([store, 
                         dcc.Markdown('#### Model Pipeline'), sep,
                         dcc.Markdown(f'**Step 3: {pred_type_string} model**'), 
                         scaling, redim, num_comp,
                         sep, model,
                         alpha, cvalue, max_depth, min_sample_split,
                         min_samples_leaf, max_features, max_leaf_nodes,
                         sep, buttons])
    
    code_string = '  '
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([dcc.Markdown('#### Preview'), sep,
                         code_box(code_string, id), error, sep,
                         html.Div([], id=f'{id}_diagram')])
    
    @callback(Output(f'{id}_nc', 'value'),
              Output(f'{id}_nc_div', 'style'),
              Input(f'{id}_pca_switch', 'value'))
    def update_nc_style(pca_switch):

        if len(pca_switch) > 0:
            return None, show
        else:
            if 'pca' in steps:
                steps.pop('pca')
            return None, hide
    
    @callback(Output(f'{id}_alpha_div', 'style'),
              Output(f'{id}_cvalue_div', 'style'),
              Output(f'{id}_max_depth_div', 'style'),
              Output(f'{id}_min_sample_split_div', 'style'),
              Output(f'{id}_min_samples_leaf_div', 'style'),
              Output(f'{id}_max_features_div', 'style'),
              Output(f'{id}_max_leaf_nodes_div', 'style'),
              Input(f'{id}_model', 'value'))
    def update_hyperparameters(model):

        if model == 'LinearRegression':
            return hide, hide, hide, hide, hide, hide, hide
        elif model == 'Ridge':
            return show, hide, hide, hide, hide, hide, hide
        elif model == 'Lasso':
            return show, hide, hide, hide, hide, hide, hide
        elif model in ['DecisionTreeRegressor', 'DecisionTreeClassifier']:
            return hide, hide, show, show, show, hide, show
        elif model in ['RandomForestRegressor', 'RandomForestClassifier']:
            return hide, hide, show, show, show, show, show
        elif model == 'LogisticRegression':
            return hide, show, hide, hide, hide, hide, hide
        else:
            return hide, hide, hide, hide, hide, hide, hide

    @callback(Output(f'{id}_diagram', 'children'),
              Output(f'{id}_code', 'children'),
              Output('reg_params', 'data'),
              Input(f'{id}_scaling', 'value'),
              Input(f'{id}_nc', 'value'),
              Input(f'{id}_model', 'value'),
              Input(f'{id}_alpha', 'value'),
              Input(f'{id}_cvalue', 'value'),
              Input(f'{id}_max_depth', 'value'),
              Input(f'{id}_min_sample_split', 'value'),
              Input(f'{id}_min_samples_leaf', 'value'),
              Input(f'{id}_max_features', 'value'),
              Input(f'{id}_max_leaf_nodes', 'value'))
    def update_regression_pipeline(scaling, nc, model, alpha, cvalue,
                                   max_depth, min_sample_split, min_samples_leaf,
                                   max_features, max_leaf_nodes):

        params = {}
        if scaling == 'StandardScaler':
            steps['scaling'] = StandardScaler()
        elif scaling == 'Normalizer':
            steps['scaling'] = Normalizer()
        if nc is not None:
            steps['pca'] = PCA()
            if nc is not None:
                params['pca__n_components'] = nc
        iters = 'max_iter=1000000' if model in ['Ridge', 'Lasso', 'LogisticRegression'] else ''
        rand = 'random_state=0' if 'DecisionTree' in model or 'RandomForest' in model else ''
        steps['model'] = eval(f'{model}({iters}{rand})')
        
        pipe_steps = []
        keys = ['dummy', 'scaling', 'pca', 'model']
        for key in keys:
            if key in steps:
                pipe_steps.append((key, steps[key]))
            
        pipe = Pipeline(pipe_steps)

        if model in ['Ridge', 'Lasso']:
            if alpha is not None:
                params['model__alpha'] = alpha
        if model == 'LogisticRegression':
            if cvalue is not None:
                params['model__C'] = cvalue
        if model in ['DecisionTreeRegressor', 'RandomForestRegressor',
                     'DecisionTreeClassifier', 'RandomForestClassifier']:
            if max_depth is not None:
                params['model__max_depth'] = max_depth
            if min_sample_split is not None:
                params['model__min_sample_split'] = min_sample_split
            if min_samples_leaf is not None:
                params['model__min_samples_leaf'] = min_samples_leaf
            if max_leaf_nodes is not None:
                params['model__max_leaf_nodes'] = max_leaf_nodes
        if model in ['RandomForestRegressor', 'RandomForestClassifier']:
            if max_features is not None:
                params['model__max_features'] = max_features 

        diagram = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(pipe._repr_html_())
        code = pred_reg_model_code(steps, params)
        return diagram, code, params

    @callback(Output('pred_reg_model_div', 'style', allow_duplicate=True),
              Output('pred_cats_div', 'style', allow_duplicate=True),
              Input(f'{id}_back', 'n_clicks'),
              prevent_initial_call=True)
    def update_back_button(n_clicks):

        if 'scaling' in steps:
            steps.pop('scaling')
        if 'model' in steps:
            steps.pop('model')
        if n_clicks is None:
            return show, hide
        else:
            return hide, show
    
    @callback(Output('pred_reg_model_div', 'style', allow_duplicate=True),
              Output('pred_fitting_div', 'children', allow_duplicate=True),
              Output('pred_fitting_div', 'style', allow_duplicate=True),
              Input(f'{id}_next', 'n_clicks'),
              State('current', 'data'),
              prevent_initial_call=True)
    def update_next_button(n_clicks, current):

        if n_clicks is None:
            return show, [], hide
        else:
            controls, previews = pred_model_fit(current, nodes, steps, id='pred_fitting')
            page = two_columns(controls, previews, id='pred_fitting_page')
            return hide, page, show
 
    return controls, previews


def pred_model_fit(current, nodes, steps, id):

    num_folds = sidebar_input('Fold number', 'number', id=f'{id}_num_folds',
                              min=2, max=20, step=1, value=5)
    test_switch = sidebar_switch('Test dataset', 'no/yes', id=f'{id}_test_switch')
    test_ratio = sidebar_input('Test set ratio', 'number', id=f'{id}_test_ratio',
                              min=0.1, max=0.5, step=0.05, value=0.30)
    fit = dbc.Row([std_button('Fit model', id=f'{id}_fit', width=120)], justify='end',
                  style={'margin-left': 3, 'margin-right': 3, 'margin-top': 0, 'margin-bottom': 5})

    buttons = dbc.Row([std_button('Back', id=f'{id}_back'),
                       std_button('Next', id=f'{id}_next')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    
    # store = []
    controls = html.Div([dcc.Store('yhat_train'), dcc.Store('yhat_test'),
                         dcc.Store('y_train'), dcc.Store('y_test'),
                         dcc.Markdown('#### Model Pipeline'), sep,
                         dcc.Markdown('**Step 4: model fitting**'), 
                         num_folds, test_switch, test_ratio, fit,
                         sep, buttons])
    
    code_string = '  '
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    previews = html.Div([dcc.Markdown('#### Preview'), sep,
                         code_box(code_string, id), error, sep,
                         dcc.Markdown('', id=f'{id}_summary', style={'margin-top': 10}),
                         html.Div([], id=f'{id}_scores', style={'margin-top': 25})])
    
    @callback(Output(f'{id}_test_ratio_div', 'style'),
              Input(f'{id}_test_switch', 'value'))
    def update_ratio_style(test_switch):

        if len(test_switch) > 0:
            return show
        else:
            return hide
    
    @callback(Output(f'{id}_scores', 'children'),
              Output(f'{id}_summary', 'children'),
              Output(f'{id}_error', 'children'),
              Output(f'{id}_next', 'disabled'),
              Output('yhat_train', 'data'),
              Output('yhat_test', 'data'),
              #Output('idx_train', 'data'),
              #Output('idx_test', 'data'),
              Output('y_train', 'data'),
              Output('y_test', 'data'),
              # Output(f'{id}_code', 'children'),
              Input(f'{id}_fit', 'n_clicks'),
              State(f'{id}_num_folds', 'value'),
              State(f'{id}_test_switch', 'value'),
              State(f'{id}_test_ratio', 'value'),
              State('pred_vars_predicted', 'value'),
              State('pred_vars_predictors', 'value'),
              State('reg_params', 'data'),
              State('pred_vars_pred_type', 'data'),
              State('current', 'data'))
    def update_fit_scores(n_clicks, num_folds, test_switch, ratio,
                          predicted, predictors, params, pred_type, current):

        if n_clicks is None:
            return [], '', '', True, None, None, None, None
        else:
            cv = KFold(num_folds, shuffle=True, random_state=0)
            parameters = {key: eval(value) for key, value in params.items()}
            order = ['dummy', 'scaling', 'pca', 'model']
            sorted_steps = [(name, steps[name]) for name in order if name in steps]
            pipe = Pipeline(sorted_steps)

            idx = current['idx']
            input_name, input_data = nodes[idx].content
            all_columns = index_labels(input_data.columns)
            x = input_data[all_columns[predictors]]
            y = input_data[all_columns[predicted]]
            if pred_type == 'R':
                score_param = {}
                method_param = {}
            else:
                score_param = {'scoring': 'roc_auc'}
                method_param = {'method': 'predict_proba'}
            if len(test_switch) > 0:
                datasets = train_test_split(x.reset_index(drop=True), y,
                                            test_size=ratio, random_state=0)
                x_train, x_test, y_train, y_test = datasets
            else:
                x_train, y_train, y_test = x, y, []
            try:
                if len(params) > 0:
                    search = GridSearchCV(pipe, parameters, cv=cv, n_jobs=-1, **score_param)
                    search.fit(x_train, y_train)
                    best_index = search.best_index_
                    scores = np.array([search.cv_results_[f'split{i}_test_score'][best_index]
                                       for i in range(num_folds)])
                    best_params = search.best_params_
                    param_string = '\n'.join([f"- {p[p.index('__')+2:]}: {best_params[p]}"
                                              for p in params])
                    summary = ('Best parameters:\n'
                               f'{param_string}\n\n'
                               f'Cross-validation score: {scores.mean().round(4)}\n')
                    model = search.best_estimator_
                else:
                    scores = cross_val_score(pipe, x_train, y_train, cv=cv, **score_param)
                    summary = f'Cross-validation score: {scores.mean().round(4)}\n'
                    model = pipe

                model.fit(x_train, y_train)
                if len(test_switch) > 0:
                    test_score = model.score(x_test, y_test, **score_param)
                    summary += f'Test score: {np.round(test_score, 4)}\n'
                
                yhat_train = cross_val_predict(model, x_train, y_train, cv=cv, **method_param)
                if len(y_test) > 0:
                    model.fit(x_train, y_train)
                    if pred_type == 'R':
                        yhat_test = model.predict(x_test)
                    else:
                        yhat_test = model.predict_proba(x_test)
                else:
                    yhat_test = []
                store_data = yhat_train, yhat_test, y_train, y_test
                
                index = [f'fold{i}' for i in range(num_folds)]
                score_name = 'R-squared' if pred_type == 'R' else 'AUC'
                table = pd.DataFrame({f'{score_name}': scores.round(4)}, index=index).T
                data_table = data_frame(table, entire=False)
                summary_markdown = f'```\n{summary}```'
                
                return (data_table, summary_markdown, '', False) + store_data 
            except Exception as err:
                return [], '', f'\n\n**`Error`**`: {err}`', True, None, None, None, None
    
    @callback(Output(f'{id}_code', 'children'),
              Input(f'{id}_num_folds', 'value'),
              Input(f'{id}_test_switch', 'value'),
              Input(f'{id}_test_ratio', 'value'),
              State('reg_params', 'data'),
              State('pred_vars_pred_type', 'data'))
    def update_code(num_folds, test_switch, ratio, params, pred_type):

        return pred_fit_code(params, num_folds, test_switch, ratio, pred_type)
    
    @callback(Output('pred_fitting_div', 'style', allow_duplicate=True),
              Output('pred_reg_model_div', 'style', allow_duplicate=True),
              Input(f'{id}_back', 'n_clicks'),
              prevent_initial_call=True)
    def update_back_button(n_clicks):

        if n_clicks is None:
            return show, hide
        else:
            return hide, show
    
    @callback(Output('pred_fitting_div', 'style', allow_duplicate=True),
              Output('pred_reg_analysis_div', 'children', allow_duplicate=True),
              Output('pred_cls_analysis_div', 'children', allow_duplicate=True),
              Output('pred_reg_analysis_div', 'style', allow_duplicate=True),
              Output('pred_cls_analysis_div', 'style', allow_duplicate=True),
              Input(f'{id}_next', 'n_clicks'),
              State('pred_vars_pred_type', 'data'),
              State('y_train', 'data'),
              State('y_test', 'data'),
              State('current', 'data'),
              prevent_initial_call=True)
    def update_next_button(n_clicks, pred_type, y_train, y_test, current):

        if n_clicks is None:
            return show, [], [], hide, hide
        else:
            if pred_type == 'R':
                controls, previews = pred_reg_analysis(current, nodes, id='pred_reg_analysis')
                page = two_columns(controls, previews, id='pred_reg_analysis_page')
                return hide, page, [], show, hide
            else:
                controls, previews = pred_cls_analysis(nodes, y_train, y_test, id='pred_cls_analysis')
                page = two_columns(controls, previews, id='pred_cls_analysis_page')
                return hide, [], page, hide, show
    
    return controls, previews


def pred_reg_analysis(current, nodes, id):

    ptype_options = ['prediction plot', 'residual plot']
    ptype = sidebar_dropdown('Types of plot', ptype_options, None, id=f'{id}_ptype')
    # std_resid_switch = sidebar_switch('Std. residual', 'no/yes', id=f'{id}_std_resid')
    
    buttons = dbc.Row([std_button('Back', id=f'{id}_back'),
                       std_button('Save', id=f'{id}_save')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    
    controls = html.Div([dcc.Markdown('#### Model Pipeline'), sep,
                         dcc.Markdown('**Step 5: prediction analysis**'),
                         ptype, sep, buttons])
    
    code_string = '  '
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    figure = go.Figure(go.Bar(x=[], y=[]))
    previews = html.Div([dcc.Markdown('#### Preview'), sep,
                         code_box(code_string, id), error, sep,
                         dcc.Graph(figure=figure, id=f'{id}_pred_graph', style=hide),
                         dcc.Graph(figure=figure, id=f'{id}_resid_graph', style=hide)])

    @callback(Output(f'{id}_pred_graph', 'figure'),
              Output(f'{id}_resid_graph', 'figure'),
              Output(f'{id}_pred_graph', 'style'),
              Output(f'{id}_resid_graph', 'style'),
              Output(f'{id}_code', 'children'),
              Input(f'{id}_ptype', 'value'),
              # Input(f'{id}_std_resid', 'value'),
              State('pred_vars_predicted', 'value'),
              State('pred_fitting_test_switch', 'value'),
              State('yhat_train', 'data'),
              State('yhat_test', 'data'),
              State('y_train', 'data'),
              State('y_test', 'data'),
              State('current', 'data'))
    def update_pred_resid_plots(ptype, predicted, test_switch,
                                yhat_train, yhat_test, y_train, y_test, current):

        code = pred_reg_analysis_code(ptype, test_switch)

        null_figure = go.Figure(go.Scatter(x=[], y=[]))
        if ptype is None:
            return null_figure, null_figure, hide, hide, code
        
        idx = current['idx']
        input_data = nodes[idx].content[1]
        column_labels = index_labels(input_data.columns)
        y = input_data[column_labels[predicted]].values
        y_train, y_test = np.array(y_train), np.array(y_test)
        yhat_train, yhat_test = np.array(yhat_train), np.array(yhat_test)

        resid_train, resid_test = y_train - yhat_train, y_test - yhat_test
        
        xmin = np.array(yhat_train).min()
        xmax = np.array(yhat_train).max()
        if len(y_test) > 0:
            xmin = min([xmin, np.array(yhat_test).min()])
            xmax = max([xmax, np.array(yhat_test).max()])
        if ptype == 'prediction plot':
            pred_style = show
            xxmin = min([xmin, y.min()])
            xxmax = max([xmax, y.max()])
            scatters = [go.Scatter(x=yhat_train, y=y_train, mode='markers', 
                                   marker=dict(color='rgba(255,255,255,0)', size=6,
                                               line=dict(color='rgba(0,0,255,0.35)', width=2)),
                                   name='Cross-validation', showlegend=len(y_test) > 0),
                        go.Scatter(x=yhat_test, y=y_test, mode='markers', 
                                   marker=dict(color='rgba(255,255,255,0)', size=6,
                                               line=dict(color='rgba(255,0,0,0.35)', width=2)),
                                   name='Test'),
                        go.Scatter(x=[xxmin, xxmax], y=[xxmin, xxmax], mode='lines',
                                   line=dict(color='black', dash='dash'), showlegend=False)]
            pred_figure = go.Figure(scatters)
            if len(y_test) > 0:
                pred_figure.update_layout(legend=dict(yanchor='top', xanchor='left',
                                                      x=1.02, y=1.0,
                                                      bordercolor='Black', borderwidth=1))
                fig_width = 500
            else:
                fig_width = 360
            pred_figure.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                                      plot_bgcolor='white', width=fig_width, height=360)
            pred_figure.update_layout(xaxis_title='Predicted values', yaxis_title='Actual values')
            pred_figure.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            pred_figure.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            pred_figure.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
            pred_figure.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
        else:
            pred_style = hide
            pred_figure = null_figure
        if ptype == 'residual plot':
            resid_style = show
            scatters = [go.Scatter(x=yhat_train, y=resid_train, mode='markers', 
                                   marker=dict(color='rgba(255,255,255,0)', size=6,
                                               line=dict(color='rgba(0,0,255,0.35)', width=2)),
                                   name='Cross-validation', showlegend=len(y_test) > 0),
                        go.Scatter(x=yhat_test, y=resid_test, mode='markers', 
                                   marker=dict(color='rgba(255,255,255,0)', size=6,
                                               line=dict(color='rgba(255,0,0,0.35)', width=2)),
                                   name='Test'), 
                        go.Scatter(x=[xmin, xmax], y=[0, 0], mode='lines',
                                   line=dict(color='black', dash='dash'), showlegend=False)]
            resid_figure = go.Figure(scatters)
            if len(y_test) > 0:
                resid_figure.update_layout(legend=dict(yanchor='top', xanchor='left',
                                                       x=1.02, y=1.0,
                                                       bordercolor='Black', borderwidth=1))
                fig_width = 500
            else:
                fig_width = 360
            resid_figure.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                                       plot_bgcolor='white', width=fig_width, height=360)
            resid_figure.update_layout(xaxis_title='Predicted values', yaxis_title='Residuals')
            resid_figure.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            resid_figure.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            resid_figure.update_xaxes(showgrid=True, gridwidth=0.5,
                                      gridcolor='black', griddash='dot')
            resid_figure.update_yaxes(showgrid=True, gridwidth=0.5,
                                      gridcolor='black', griddash='dot')
        else:
            resid_style = hide
            resid_figure = null_figure

        return pred_figure, resid_figure, pred_style, resid_style, code
            
    @callback(Output('pred_reg_analysis_div', 'style', allow_duplicate=True),
              Output('pred_fitting_div', 'style', allow_duplicate=True),
              Input(f'{id}_back', 'n_clicks'),
              prevent_initial_call=True)
    def update_back_button(n_clicks):

        if n_clicks is None:
            return show, hide
        else:
            return hide, show

    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_save', 'n_clicks'),
              State('blueprint', 'figure'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'pred_fitting_summary', 'children'),
              State('pred_reg_model_model', 'value'),
              State(f'pred_vars_code', 'children'),
              State(f'pred_cats_code', 'children'),
              State(f'pred_reg_model_code', 'children'),
              State(f'pred_fitting_code', 'children'),
              State(f'pred_reg_analysis_code', 'children'),
              prevent_initial_call=True)
    def update_exp_save_button(n_clicks, figure, tabs, current, summary, model,
                               code1, code2, code3, code4, code5):
        
        code_markdown = '\n'.join([code1, code2, code3, code4, code5])
        code = code_markdown.replace('```python\n', '').replace('\n```', '')

        idx = current['idx']
        input_name = nodes[idx].content[0]

        summary = summary.replace('```\n', '').replace('\n```', '')
        sep = '=' * 40
        summary = f'{model} model:\n{sep}\n{summary}\n{sep}'
        fit = 'regression'
        model_info = f'Predictive {fit} model fit on <b>{input_name}</b>'
        content = input_name, summary, model_info

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'model')

    return controls, previews


def pred_cls_analysis(nodes, y_train, y_test, id):

    y_train = pd.Series(y_train).astype(str)
    classes = y_train.unique()
    classes.sort()
    cat = sidebar_dropdown('Select a category', classes, classes[-1],
                           id=f'{id}_cat', clearable=False)
    test_switch = sidebar_switch('Show test set', 'no/yes',
                                 id=f'{id}_test_switch', disabled=len(y_test)==0)
    threshold = sidebar_input('Threshold', 'number', id=f'{id}_threshold',
                              min=0.002, max=0.998, step=0.002, value=0.50)
    buttons = dbc.Row([std_button('Back', id=f'{id}_back'),
                       std_button('Save', id=f'{id}_save')],
                      justify='between', style={'margin-left': 3, 'margin-right': 3})
    
    controls = html.Div([dcc.Markdown('#### Model Pipeline'), sep,
                         dcc.Markdown('**Step 5: prediction analysis**'),
                         cat, test_switch, threshold, sep, buttons])
    
    code_string = '  '
    error = dcc.Markdown('', id=f'{id}_error', style={'color': 'red', 'margin-top': 15})
    figure = go.Figure(go.Bar(x=[], y=[]))
    results = html.Div([dcc.Graph(figure=figure, id=f'{id}_graph',
                                  style={'display': 'inline-block',
                                         'verticalAlign': 'top', 
                                         'width': 440}),
                        html.Div([], id=f'{id}_confusion', 
                                 style={'display': 'inline-block',
                                        'verticalAlign': 'top', 
                                        #'margin-top':35, 
                                        'width': 320})])
    previews = html.Div([dcc.Markdown('#### Preview'), sep,
                         code_box(code_string, id), error, sep,
                         results,
                         dcc.Markdown('', id=f'{id}_summary')])
    
    @callback(Output(f'{id}_graph', 'figure'),
              Output(f'{id}_confusion', 'children'),
              Output(f'{id}_code', 'children'),
              Input(f'{id}_cat', 'value'),
              Input(f'{id}_test_switch', 'value'),
              Input(f'{id}_threshold', 'value'),
              State('yhat_train', 'data'),
              State('yhat_test', 'data'),
              State('y_train', 'data'),
              State('y_test', 'data'))
    def update_auc_plot(cat, test_switch, threshold, yhat_train, yhat_test, y_train, y_test):

        if yhat_train is None:
            return go.Figure(go.Scatter(x=[], y=[])), [], ''

        y_train = pd.Series(y_train)
        #classes_values = y_train.unique()
        if is_bool_dtype(y_train):
            classes_values = [False, True]
            classes = ['False', 'True']
            cat_value = eval(cat)
            y_train = y_train.astype(str)
        else:
        #    y_num = y_train
        #    cat_dict = {str(c): c for c in y_num.unique()}
        #    cat_value = cat_dict[cat]
        #    y_train = y_train.astype(str)
        #else:
        #    cat_value = cat
            classes = y_train.unique()
            classes.sort()
            classes_values = classes
            cat_value = cat
        proba_train = pd.DataFrame(yhat_train, columns=classes)
        
        fpr, tpr, thrds = roc_curve(y_train==cat, proba_train[cat])
        if threshold is None:
            threshold = 0.5
        idx = np.argmin(abs(thrds - threshold))
        plots = [go.Scatter(x=fpr, y=tpr, mode='lines',
                            line=dict(color='Blue', width=2), name='Cross-validation'),
                 go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                            line=dict(color='Black', width=1.5, dash='dash'), showlegend=False),
                 go.Scatter(x=fpr[[idx]], y=tpr[[idx]], mode='markers',
                            marker=dict(color='Blue', size=12, opacity=0.6), showlegend=False)]
        ypred_train = proba_train[cat] > threshold
        conf_train = pd.DataFrame(confusion_matrix(y_train==cat, ypred_train, normalize='true'),
                                  index=['Actual false', 'Actual true'],
                                  columns=['Predicted false', 'Predicted true']).round(4)
        confusions = [dcc.Markdown('```Cross-validation:```')]
        confusions.extend(data_frame(conf_train, entire=False))
        summary = pd.DataFrame({'Accuracy': [(ypred_train == (y_train==cat)).mean()],
                                'TPR + TNR': [conf_train.iloc[0, 0] + conf_train.iloc[1, 1]],
                                'f1-score': [f1_score(y_train==cat, ypred_train)]}).round(4)
        confusions.append(html.Div(data_frame(summary, entire=False), style={'margin-top': 8}))

        if len(test_switch) > 0:
            y_test = pd.Series(y_test).astype(str)
            proba_test = pd.DataFrame(yhat_test, columns=classes)
            fpr_test, tpr_test, thrds_test = roc_curve(y_test==cat, proba_test[cat])
            plots.append(go.Scatter(x=fpr_test, y=tpr_test, mode='lines',
                                    line=dict(color='Red', width=2), name='Test'))
            
            ypred_test = proba_test[cat] > threshold
            conf_test = pd.DataFrame(confusion_matrix(y_test==cat, ypred_test, normalize='true'),
                                     index=['Actual false', 'Actual true'],
                                     columns=['Predicted false', 'Predicted true']).round(4)
            confusions.append(dcc.Markdown('``` ```'))
            confusions.append(dcc.Markdown('\n```Test```'))
            confusions.extend(data_frame(conf_test, entire=False))
            summary = pd.DataFrame({'Accuracy': [(ypred_test == (y_test==cat)).mean()],
                                    'TPR + TNR': [conf_test.iloc[0, 0] + conf_test.iloc[1, 1]],
                                    'f1-score': [f1_score(y_test==cat, ypred_test)]}).round(4)
            confusions.append(html.Div(data_frame(summary, entire=False), style={'margin-top': 8}))
        
        figure = go.Figure(plots)
        figure.update_layout(margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor='white',
                             width=430, height=430)
        figure.update_xaxes(title='False positive rate')
        figure.update_yaxes(title='True positive rate')
        figure.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        figure.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        figure.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
        figure.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black', griddash='dot')
        figure.update_layout(legend=dict(yanchor='bottom', xanchor='right', x=0.99, y=0.01,
                                         bordercolor='Black', borderwidth=1))

        code = pred_cls_analysis_code(classes_values, cat_value, test_switch, threshold)

        return figure, confusions, code
    
    @callback(Output('pred_cls_analysis_div', 'style', allow_duplicate=True),
              Output('pred_fitting_div', 'style', allow_duplicate=True),
              Input(f'{id}_back', 'n_clicks'),
              prevent_initial_call=True)
    def update_back_button(n_clicks):

        if n_clicks is None:
            return show, hide
        else:
            return hide, show
    
    @callback(Output('blueprint', 'figure', allow_duplicate=True),
              Output('home_tabs', 'value', allow_duplicate=True),
              Output('blueprint', 'selectedData', allow_duplicate=True),
              Input(f'{id}_save', 'n_clicks'),
              State('blueprint', 'figure'),
              State('home_tabs', 'value'),
              State('current', 'data'),
              State(f'pred_fitting_summary', 'children'),
              State('pred_reg_model_model', 'value'),
              State(f'pred_vars_code', 'children'),
              State(f'pred_cats_code', 'children'),
              State(f'pred_reg_model_code', 'children'),
              State(f'pred_fitting_code', 'children'),
              State(f'pred_cls_analysis_code', 'children'),
              prevent_initial_call=True)
    def update_pred_save_button(n_clicks, figure, tabs, current, summary, model,
                                code1, code2, code3, code4, code5):
        
        code_markdown = '\n'.join([code1, code2, code3, code4, code5])
        code = code_markdown.replace('```python\n', '').replace('\n```', '')

        idx = current['idx']
        input_name = nodes[idx].content[0]

        summary = summary.replace('```\n', '').replace('\n```', '')
        sep = '=' * 40
        summary = f'{model} model:\n{sep}\n{summary}\n{sep}'
        fit = 'regression'
        model_info = f'Predictive {fit} model fit on <b>{input_name}</b>'
        content = input_name, summary, model_info

        return new_node(figure, tabs, nodes, n_clicks, content, code, current, 'model')

    return controls, previews

def pred_pipeline(current, nodes, steps):

    mv_controls, mv_previews = model_variables(current, nodes, steps, id='pred_vars')
    mv_step = html.Div(two_columns(mv_controls, mv_previews, id='pred_vars_page'),
                       id='pred_vars_div', style=hide)
    
    mc_controls, mc_previews = pred_cats(current, nodes, steps, id='pred_cats')
    mc_step = html.Div(two_columns(mc_controls, mc_previews, id='pred_cats_page'),
                       id='pred_cats_div', style=hide)
    
    ms_controls, ms_previews = pred_regression_model(nodes, steps, 'R', id='pred_reg_model')
    ms_step = html.Div(two_columns(ms_controls, ms_previews, id='pred_reg_model_page'),
                        id='pred_reg_model_div', style=hide)
    
    mf_controls, mf_previews = pred_model_fit(current, nodes, steps, id='pred_fitting')
    mf_step = html.Div(two_columns(mf_controls, mf_previews, id='pred_fitting_page'),
                       id='pred_fitting_div', style=hide)
    
    rma_controls, rma_previews = pred_reg_analysis(current, nodes, id='pred_reg_analysis')
    rma_step = html.Div(two_columns(rma_controls, rma_previews, id='pred_reg_analysis_page'),
                        id='pred_reg_analysis_div', style=hide)
    
    y_train = [0]
    y_test = []
    cma_controls, cma_previews = pred_cls_analysis(nodes, y_train, y_test, id='pred_cls_analysis')
    cma_step = html.Div(two_columns(cma_controls, cma_previews, id='pred_cls_analysis_page'),
                        id='pred_cls_analysis_div', style=hide)

    return [mv_step, mc_step, ms_step, mf_step, rma_step, cma_step]


def home(data):

    current_node = PSNode([1.0, 8.5], 'data', None, content={'data': data})
    nodes = [current_node]
    steps = {}
    current = {'name': 'data', 'idx': 0}

    # title = [dcc.Markdown("## Panda Shifu", style={'margin-top': 25}), sep]
    title = [dcc.Markdown(" ", style={'margin-top': 25})]

    all_tabs = [dcc.Tab(id='blueprint_tab', value='bp_tab', label='Flow',
                        style=tab_style, selected_style=tab_style),
                dcc.Tab(id='dataframe_tab', value='df_tab', label='Operations',
                        style=tab_style, selected_style=tab_style, disabled_style=tab_style),
                dcc.Tab(id='visual_tab', value='dv_tab', label='Visuals',
                        style=tab_style, selected_style=tab_style, disabled_style=tab_style),
                dcc.Tab(id='model_tab', value='md_tab', label='Model',
                        style=tab_style, selected_style=tab_style, disabled_style=tab_style),
                dcc.Tab(id='pygwalker_tab', value='pg_tab', label='PyGWalker',
                        style=tab_style, selected_style=tab_style, disabled_style=tab_style),
                dcc.Tab(id='about_tab', value='ab_tab', label='About',
                        style=tab_style, selected_style=tab_style, disabled_style=tab_style)]
    page_tabs = [dcc.Tabs(all_tabs, id="home_tabs", value='df_tab', style=tabs_style), sep]

    all_buttons = [std_button('Export', id='export_button')]
    home_buttons = html.Div(all_buttons, id='home_buttons', style=buttons_style)

    bp_view, nodes = blueprint(id='bp_view', data=data, display=True)

    operations = ['Select columns', 'Filter rows', 'Treat missing values',
                  'Sort', 'Group by', 'Pivot table', 'Add a column']
    df_ops = sidebar_dropdown('Type of operations', operations, None, id='df_ops')
    df_components = [op_none(current, nodes, id='none'),
                     select_columns(current, nodes, id='columns'),
                     filter_rows(current, nodes, id='filter'),
                     treat_na(current, nodes, id='treat_na'),
                     sort_rows(current, nodes, id='sorts'),
                     grouped(current, nodes, id='grouped'),
                     pivot(current, nodes, id='pivot'),
                     add_column(current, nodes, id='add_column')]
    df_sidebar = [dcc.Markdown('#### Data Operations'), sep, df_ops,
                  html.Div([c[0] for c in df_components], 'df_controls', style=hide)]
    df_preview = [dcc.Markdown('#### Preview'), sep,
                  html.Div([c[1] for c in df_components], 'df_previews', style=hide)]
    df_page = [two_columns(df_sidebar, df_preview, id='df_page'),
               html.Div([home_buttons, bp_view], id='bp_view_div', style=hide)]

    pygw_html = html.Div([], id="pygw_html")
    pygw_page = [html.Div([pygw_html], id='pygw_page', style=hide)]

    analysis = ['Univariate distribution', 'Bar chart', 'Scatterplot', 'Line plot']
    vd_components = [univar_visual(current, nodes, id='univariate'),
                     bar_visual(current, nodes, id='barchart'),
                     scatter_visual(current, nodes, id='scatterplot'),
                     lineplot_visual(current, nodes, id='lineplot')]
    visual_type = sidebar_dropdown('Type of analysis', analysis, None, id='dv_type')
    visual_sidebar = [dcc.Markdown('#### Data Visualization'), sep,
                      visual_type,
                      html.Div([c[0] for c in vd_components], id='dv_controls')]
    visual_preview = [dcc.Markdown('#### Preview'), sep,
                      html.Div([c[1] for c in vd_components], id='dv_previews')]
    visual_page = [two_columns(visual_sidebar, visual_preview, id='visual_page', style=hide)]

    model_type = sidebar_dropdown('Purpose of model',
                                  ['Explanatory modeling', 'Predictive modeling'], None,
                                  id='model_type')
    model_buttons = dbc.Row([std_button('Next', id='model_type_next')],
                            justify='end', style={'margin-left': 3, 'margin-right': 3})
    ms_controls = html.Div([dcc.Markdown('#### Model Pipeline'), sep, model_type, model_buttons],
                           id='model_start_controls')
    ms_previews = html.Div([dcc.Markdown('#### Preview'), sep], id='model_start_previews')
    model_start_page = [html.Div(two_columns(ms_controls, ms_previews, id='model_start_page'), id='model_start_div')]
    pred_pages = pred_pipeline(current, nodes, steps)
    exp_pages = exp_pipeline(current, nodes)
    model_page = [html.Div(model_start_page + pred_pages + exp_pages, id='model_page', style=hide)]

    about_string = ('### Introduction\n'
                    '**```PandaShifu```** is an open-source Python package that provides friendly user interfaces for descriptive ' 
                    'and predictive analytics on a given dataset. '
                    'Specifically, the package is capable of processing and visualizing the given dataset, '
                    'and building pipelines for econometrical and machine learning models. '
                    'The software is developed to facilitate the teaching of the following courses offered by NUS Business School:\n'
                    '- [**DAO2702/DAO2702X Programming for Business Analytics**](https://nusmods.com/courses/DAO2702/programming-for-business-analytics)\n'
                    '- [**BMK5202 Python Programming for Business Analytics**](https://nusmods.com/courses/BMK5202/python-programming-for-business-analytics)\n'
                    '- [**BMH5104 Artificial Intelligence for HR**](https://nusmods.com/courses/BMH5104/artificial-intelligence-for-hr)\n\n'
                    '### Installation and Source\n'
                    'The **```PandaShifu```** package can be installed from the [PyPI](https://pypi.org/project/pandashifu/) platform via the command:\n'
                    '```\npip install pandashifu\n```\n'
                    'The source code of the package is hosted at [GitHub](https://github.com/XiongPengNUS/PandaShifu)\n\n'
                    '### Author\n'
                    'The **```Panda Shifu```** package is developed and maintained by Dr. Xiong Peng, who is currently a senior lectuerer at the NUS Business School.\n')
    about_page = [html.Div(dcc.Markdown(about_string, style={'margin-left': 120, 'margin-right': 120, 'margin-top': 50}),
                           id='about_page', style=hide)]

    home_page = [html.Div(page_tabs + df_page + pygw_page + visual_page + model_page + about_page, id='home_page')]

    debug = [dcc.Markdown('', id='debug_info')]

    store = [dcc.Store(id='current', data=current),
             dcc.Store(id='clear_model', data=False)]

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div(store + title + home_page + debug,
                          style=theme_style)

    @callback(Output('pygw_page', 'style'),
              Output('pygw_html', 'children'),
              Input('pygw_page', 'style'),
              State('current', 'data'),
              prevent_initial_call=True)
    def update_pygw(style, current):

        if style['display'] == 'none':
            return hide, []
        else:
            idx = current['idx']
            dataframe = nodes[idx].content[1]
            dataframe.columns = index_labels(dataframe.columns).index
            if not isinstance(dataframe.index, pd.RangeIndex):
                dataframe.reset_index(inplace=True)

            walker = pyg.walk(dataframe, spec="./viz-code.json", use_kernel_calc=False)
            html_code = walker.to_html()
            pygw_output = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(html_code)

            return show, [pygw_output]

    @callback(Output('bp_view_div', 'style', allow_duplicate=True),
              Output('df_page', 'style', allow_duplicate=True),
              Output('pygw_page', 'style', allow_duplicate=True),
              Output('visual_page', 'style', allow_duplicate=True),
              Output('model_page', 'style', allow_duplicate=True),
              Output('about_page', 'style', allow_duplicate=True),
              Input('home_tabs', 'value'),
              prevent_initial_call=True)
    def update_select_tab(tab):

        if tab == 'bp_tab':
            return show, hide, hide, hide, hide, hide
        elif tab == 'df_tab':
            return hide, show, hide, hide, hide, hide
        elif tab == 'pg_tab':
            return hide, hide, show, hide, hide, hide
        elif tab == 'dv_tab':
            return hide, hide, hide, show, hide, hide
        elif tab == 'md_tab':
            return hide, hide, hide, hide, show, hide
        elif tab == 'ab_tab':
            return hide, hide, hide, hide, hide, show

    @callback(Output('df_controls', 'children'),
              Output('df_previews', 'children'),
              Output('df_controls', 'style'),
              Output('df_previews', 'style'),
              Input('df_ops', 'value'),
              State('current', 'data'))
    def update_df_ops(operation, current):

        idx = current['idx']
        if idx is None:
            return [], [], hide, hide

        if nodes[idx].ntype != 'data':
            return [], [], hide, hide

        if operation is None:
            controls, previews = op_none(current, nodes, id='none')
        elif operation == 'Select columns':
            controls, previews = select_columns(current, nodes, id='columns')
        elif operation == 'Filter rows':
            controls, previews = filter_rows(current, nodes, id='filter')
        elif operation == 'Treat missing values':
            controls, previews = treat_na(current, nodes, id='treat_na')
        elif operation == 'Sort':
            controls, previews = sort_rows(current, nodes, id='sorts')
        elif operation == 'Group by':
            controls, previews = grouped(current, nodes, id='grouped')
        elif operation == 'Pivot table':
            controls, previews = pivot(current, nodes, id='pivot')
        elif operation == 'Add a column':
            controls, previews = add_column(current, nodes, id='add_column')
        else:
            controls, previews = [], []

        return controls, previews, show, show

    @callback(Output('dv_controls', 'children'),
              Output('dv_previews', 'children'),
              Output('dv_controls', 'style'),
              Output('dv_previews', 'style'),
              Input('dv_type', 'value'),
              State('current', 'data'))
    def update_dv_type(atype, current):

        idx = current['idx']
        if idx is None:
            return [], [], hide, hide

        if nodes[idx].ntype != 'data':
            return [], [], hide, hide

        if atype is None:
            return [], [], hide, hide
        elif atype == 'Univariate distribution':
            controls, previews = univar_visual(current, nodes, id='univariate')
        elif atype == 'Bar chart':
            controls, previews = bar_visual(current, nodes, id='barchart')
        elif atype == 'Scatterplot':
            controls, previews = scatter_visual(current, nodes, id='scatterplot')
        elif atype == 'Line plot':
            controls, previews = lineplot_visual(current, nodes, id='lineplot')
        else:
            controls, previews = [], []

        return controls, previews, show, show
    
    @callback(Output('pred_vars_div', 'children'),
              Output('exp_vars_div', 'children'),
              Output('model_start_div', 'style'),
              Output('pred_vars_div', 'style', allow_duplicate=True),
              Output('exp_vars_div', 'style', allow_duplicate=True),
              #Output('debug_info', 'children'),
              Input('model_type_next', 'n_clicks'),
              State('model_type', 'value'),
              State('current', 'data'),
              State('clear_model', 'data'),
              State('pred_vars_div', 'children'),
              State('exp_vars_div', 'children'),
              State('pred_vars_div', 'style'),
              State('exp_vars_div', 'style'),
              prevent_initial_call=True)
    def update_md_types(n_clicks, mtype, current, clear_model,
                        pred_vars, exp_vars, pred_vars_style, exp_vars_style):

        idx = current['idx']
        if idx is None:
            return [], [], hide, hide, hide
        else:
            #if False: #n_clicks is None:
            #    return [], [], show, hide, hide
            if clear_model:
                if mtype == 'Predictive modeling':
                    controls, previews = model_variables(current, nodes, steps, id='pred_vars', mtype='P')
                    page =  two_columns(controls, previews, id='pred_vars_page', style=show)
                    return page, [], hide, show, hide
                elif mtype == 'Explanatory modeling':
                    controls, previews = model_variables(current, nodes, [], id='exp_vars', mtype='E')
                    page =  two_columns(controls, previews, id='pred_vars_page', style=show)
                    return [], page, hide, hide, show
                else:
                    return [], [], show, hide, hide
            else:
                return pred_vars, exp_vars, hide, pred_vars_style, exp_vars_style
    
    @callback(Output('model_type_next', 'disabled'),
              Input('model_type', 'value'))
    def update_mtype_next_disable(mtype):

        return mtype is None

    @callback(Output('model_start_div', 'style', allow_duplicate=True),
              Output('pred_vars_div', 'style', allow_duplicate=True),
              Output('pred_cats_div', 'style', allow_duplicate=True),
              Output('pred_reg_model_div', 'style', allow_duplicate=True),
              Output('pred_fitting_div', 'style', allow_duplicate=True),
              Output('pred_reg_analysis_div', 'style', allow_duplicate=True),
              Output('pred_cls_analysis_div', 'style', allow_duplicate=True),
              Output('exp_vars_div', 'style', allow_duplicate=True),
              Output('exp_fit_div', 'style', allow_duplicate=True),
              Input('clear_model', 'data'),
              State('pred_vars_div', 'style'),
              State('pred_cats_div', 'style'),
              State('pred_reg_model_div', 'style'),
              State('pred_fitting_div', 'style'),
              State('pred_reg_analysis_div', 'style'),
              State('pred_cls_analysis_div', 'style'),
              State('exp_vars_div', 'style'),
              State('exp_fit_div', 'style'),
              State('home_tabs', 'value'),
              prevent_initial_call=True)
    def update_clear_model(clear_model, 
                           pred_vars_style, pred_cats_style, pred_reg_model_style,
                           pred_fitting_style, pred_reg_analysis_style, pred_cls_analysis_style,
                           exp_vars_style, exp_fit_style, tabs):

        if clear_model:
            return show, hide, hide, hide, hide, hide, hide, hide, hide
        else:
            if tabs == 'md_tab':
                return (hide, 
                        pred_vars_style, pred_cats_style, pred_reg_model_style,
                        pred_fitting_style, pred_reg_analysis_style, pred_cls_analysis_style,
                        exp_vars_style, exp_fit_style)
            else:
                return show, hide, hide, hide, hide, hide, hide, hide, hide
                
    @callback(Input('export_button', 'n_clicks'),
              prevent_initial_call=True)
    def update_export(n_clicks):

        notebook_data = export(nodes)

        with open('code.ipynb', 'w') as f:
            json.dump(notebook_data, f)

    return app

def run(data, debug=False, tab=True, port=None):

    home_app = home(data)

    mode = 'tab' if tab else 'inline'
    port = '8050' if port is None else port
    
    home_app.run(debug=debug, jupyter_height=1300, jupyter_mode=mode, port=port)

