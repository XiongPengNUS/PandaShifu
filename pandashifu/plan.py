import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html, Input, Output, State, callback

import numpy as np
import pandas as pd

from .styles import blueprint_style, marker_style, line_style
from .styles import selected_color, unselected_color


pd.options.display.max_rows = 10
pd.options.display.max_columns = 8
pd.options.display.max_colwidth = 25

show = {'display': 'block'}
hide = {'display': 'none'}

class PSNode:

    def __init__(self, pos, ntype, before=None, levels=[], content={}, code='', imports=[]):

        self.dx = 2.5
        self.dy = -1.0

        self.pos = pos
        self.ntype = ntype
        self.before = before
        self.after = []
        self.nbranch = 0
        self.offset = None

        self.levels = levels

        self.content = content
        self.code = code
        self.imports = imports
        #self.page = None
        self.visuals = []

    def get_root(self):

        node = self.before
        if node is None:
            return self
        else:
            return node.get_root()

    def get_all(self):

        output = [self]
        for node in self.after:
            output.extend(node.get_all())

        return output

    def get_all_lines(self):

        x = []
        y = []
        for node in self.after:
            x1, x2 = self.pos[0], node.pos[0]
            y1, y2 = self.pos[1], node.pos[1]
            if x1 == x2:
                x.extend([np.nan, x1, x2, np.nan])
                y.extend([np.nan, y1, y2, np.nan])
            else:
                xmid = x1 + (x2 - x1)*np.concatenate([np.linspace(0, 0.6, 60), np.linspace(0.61, 1, 5)])
                x.extend([np.nan] + list(xmid) + [np.nan])
                ymid = y2 + (y1 - y2)/(1 + np.exp(15*(xmid - (0.65*x1 + 0.35*x2))))
                y.extend([np.nan] + list(ymid) + [np.nan])
            data = node.get_all_lines()
            x.extend(data[0])
            y.extend(data[1])

        return x, y

    def content_label(self):

        if self.ntype == 'data':
            name, data = self.content
            #data = data.fillna('NaN')
            #return f'<b>{name}</b>:<br>{data.__str__().replace('\n', '<br>')}'
            return f'<b>{name}</b>:<br>{data.shape[0]} x {data.shape[1]}'
        elif self.ntype == 'visual':
            name, _, vtype = self.content
            return f'{vtype} for <b>{name}</b>'
        elif self.ntype == 'model':
            return self.content[2]
        else:
            return 'Unknown'

    def grow(self, ntype, content, code='', imports=[]):

        x = self.pos[0] + self.dx
        y0 = self.pos[1]
        levels = self.levels + [self.nbranch]
        root = self.get_root()
        all_nodes = root.get_all()
        nlevels = len(self.levels)
        upper = []
        lower = []
        lower_nodes = []
        for node in all_nodes:
            if nlevels + 1 == len(node.levels) and nlevels > 0:
                shift = np.array(self.levels) - np.array(node.levels[:nlevels])
                nonzero_indices = np.nonzero(shift)[0]
                if len(nonzero_indices) > 0:
                    if shift[nonzero_indices[0]] > 0:
                        upper.append(node.pos[1])
                    else:
                        lower_nodes.append(node)
                        lower.append(node.pos[1])
        if upper:
            y0 = min(upper) + self.dy if y0 >= min(upper) else y0
        y = y0 + self.dy*self.nbranch
        if lower:
            ymax = max(lower)
            if y <= ymax:
                offset = y - ymax + self.dy
                for node in lower_nodes:
                    node.pos = (node.pos[0], node.pos[1] + offset)

        next_node = PSNode((x, y), ntype, self, levels, content, code=code)
        self.after.append(next_node)
        self.nbranch += 1

        return next_node


def blueprint(id, data, display=True):

    current_node = PSNode([1.0, 8.5], 'data', None, content=('data', data))
    current = {'name': current_node.content[0], 'idx': 0}
    node_label = current_node.content_label()
    nodes = [current_node]

    figure = go.Figure([go.Scatter(x=[], y=[], mode='lines',
                                   hoverinfo='skip', name="",
                                   line=line_style, showlegend=False),
                        go.Scatter(x=[1.0], y=[8.5], mode='markers',
                                   marker_size=[30], marker_symbol=['circle'],
                                   customdata=[node_label],
                                   hovertemplate="%{customdata}", name="",
                                   marker=marker_style, showlegend=False,
                                   hoverlabel=dict(font=dict(family='monospace', size=14)))])
    figure.update_layout(clickmode='event+select', hoverlabel=dict(font_size=20),
                         margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor='rgba(255,255,255,0)')
    figure.update_traces(selectedpoints=[0])
    figure.data[1].unselected.marker.opacity = 1.0
    figure.layout.hovermode = 'closest'
    figure.update_xaxes({'range': (0, 16.0), 'visible': False})
    figure.update_yaxes({'range': (0, 9.0), 'visible': False})

    current_point = {"points": [{"pointIndex": 0, "x": 1.0, "y": 8.5}]}
    graph = html.Div(dcc.Graph(figure=figure, id='blueprint', selectedData=current_point,
                      style=blueprint_style), style={'position': 'absolute'})
    view = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    view.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    thumbnail_figure = dcc.Graph(figure=view, id='thumnail', style={'width': 400, 'height': 300}, config={'staticPlot': True})
    thumbnail = html.Div(html.Div(thumbnail_figure,
                                  style={'width': 1040,
                                         'height': 585,
                                         'padding-top': 260,
                                         'padding-left': 640}), id='thumnail_div', style={'display': 'none'})

    @callback(Output('blueprint', 'figure'),
              Output('dataframe_tab', 'disabled'),
              Output('pygwalker_tab', 'disabled'),
              Output('visual_tab', 'disabled'),
              Output('model_tab', 'disabled'),
              Output('df_ops', 'value', allow_duplicate=True),
              Output('dv_type', 'value', allow_duplicate=True),
              #Output('model_type', 'value', allow_duplicate=True),
              Output('thumnail_div', 'style', allow_duplicate=True),
              Output('thumnail_div', 'children', allow_duplicate=True),
              Output('current', 'data', allow_duplicate=True),
              Output('clear_model', 'data', allow_duplicate=True),
              #Output('debug_info', 'children', allow_duplicate=True),
              Input('blueprint', 'selectedData'),
              State('blueprint', 'figure'),
              State('current', 'data'),
              State('df_ops', 'value'),
              State('dv_type', 'value'),
              #State('clear_model', 'data'),
              prevent_initial_call=True)
    def select_node_update(selectedData, figure, previous, operation, atype):

        clear_model = False
        if selectedData and selectedData['points']:
            point = selectedData['points'][0]
            s = figure['data'][1]
            idx = point['pointIndex']
            current_node = nodes[idx]
            colors = [unselected_color] * len(s['x'])
            colors[idx] = selected_color
            figure['data'][1]['marker']['color'] = colors
            current['name'] = current_node.content[0]
            current['idx'] = idx
            if previous['idx'] != current['idx']:
                operation = None
                clear_model = True
            if current_node.ntype == 'data':
                return (figure, False, False, False, False, operation, atype, hide, [],
                        current, clear_model)
            elif current_node.ntype == 'visual':
                fig_data = current_node.content[1]
                if isinstance(fig_data, list):
                    thumbnail = html.Div(fig_data,
                                         style={'width': 600,
                                                'height': 585,
                                                'padding-top': 490,
                                                'padding-left': 40})
                else:
                    view = go.Figure(fig_data)
                    w, h = fig_data['layout']['width'], fig_data['layout']['height']
                    if w > 400:
                        w, h = 400, 400/w*h
                    if h > 300:
                        w, h = 300/h*w, 300
                    view.update_layout(width=w, height=h)
                    view.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                    thumbnail_figure = dcc.Graph(figure=view, config={'staticPlot': True})
                    thumbnail = html.Div(thumbnail_figure, style={'width': 1040,
                                                                  'height': 585,
                                                                  'padding-top': 560 - h,
                                                                  'padding-left': 40})#1040 - w})
                return (figure, True, True, True, True, operation, atype, show, thumbnail,
                        current, clear_model)
            elif current_node.ntype == 'model':
                summary = current_node.content[1]
                summary_lines = summary.split('\n')
                if 'OLS' in summary_lines[0]:
                    end, offset = 12, 0
                elif 'Logit' in summary_lines[0]:
                    end, offset = 10, 20
                else:
                    end = len(summary_lines)
                    offset = (12 - end) * 12
                view = go.Figure()
                view.add_annotation(dict(font=dict(size=11, family='monospace', weight='bold'),
                                         align='left', x=0, y=0.99, showarrow=False,
                                         text='<br>'.join(summary_lines[:end]),
                                         xanchor='left', xref="paper", yref="paper"))
                view.update_layout(margin=dict(l=2, r=2, t=2, b=0),
                                   plot_bgcolor='rgba(255,255,255,0)')
                view.update_xaxes({'visible': False})
                view.update_yaxes({'visible': False})
                thumbnail_figure = dcc.Graph(figure=view, config={'staticPlot': True},
                                             style={'width': 630, 'height': 180 - offset})
                thumbnail = html.Div(thumbnail_figure,
                                     style={'width': 620,
                                            'height': 180 - offset,
                                            'padding-top': 390 + offset,
                                            'padding-left': 50})
                return (figure, True, True, True, True, operation, atype, show, thumbnail,
                        current, clear_model)
        else:
            s = figure['data'][1]
            colors = [unselected_color] * len(s['x'])
            figure['data'][1]['marker']['color'] = colors
            current['name'] = None
            current['idx'] = None
            return (figure, True, True, True, True, None, None, hide, [],
                    current, clear_model)

    style = {'display': 'block'} if display else {'display': 'none'}
    return html.Div([graph, thumbnail], id=id, style=style), nodes
