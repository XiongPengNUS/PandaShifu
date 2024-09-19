from dash import Dash, dash_table, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

import pyperclip


def std_button(label, id, inline=True, width=80, disabled=False):

    button_style = {'width': width,
                    'height': '40px',
                    'borderWidth': '2px',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin-left': '5px',
                    'margin-right': '5px',
                    'margin-top': '10px',
                    'margin-bottom': '10px',
                    'padding-left': '2px',
                    'padding-right': '2px'}
    button_style['display'] = 'inline-block' if inline else 'block'
    # button_style['display'] = 'none' if not show else button_style['display']

    button = dbc.Button(label, id=id, outline=True, color='secondary', disabled=disabled, style=button_style)

    @callback(Output(id, 'style'),
              Input(id, 'disabled'))
    def disable_style_update(disabled):

        style = button_style.copy()
        style['color'] = '#D9D9D9' if disabled else '#000000'

        return style

    return button


def sidebar_input(label, intype, id, **kwargs):

    text = dcc.Markdown(label, style={'display': 'inline-block', "width": 100, 'height': 15, 'margin-left': 5})
    item = dbc.Input(type=intype, id=id, size='sm',
                     style={"width": 100, "height": 30,
                            'display': 'inline-block',
                            'margin-left': '3%',
                            'font-family':'monospace'}, **kwargs)

    return html.Div([text, item], id=f"{id}_div",
                    style={"verticalAlign": "center", 'height': 35,
                           'margin-left': 0, 'margin-bottom': 5})


def sidebar_expr(label, id, **kwargs):

    item = dbc.Input(type='text', id=id, size='sm',
                     style={'height': 35, 'width': 205,
                            'margin-top': 5, 'margin-bottom': 10,
                            'margin-left': 5, 'margin-right': 5,
                            'font-family':'monospace'}, **kwargs)

    return html.Div([html.Div(label, style={'margin-bottom': 5, 'margin-left': 5}), item],
                    id=f"{id}_div", style={'margin-bottom': 10,
                                           'justify': 'center'})


def sidebar_inline_expr(label, id, **kwargs):

    item = dbc.Input(type='text', id=id, size='sm',
                     style={'height': 35, 'width': 100,
                            'margin-left': 5, 'margin-right': 5,
                            'display': 'inline-block',
                            'font-family':'monospace'}, **kwargs)

    return html.Div([html.Div(label, style={'margin-bottom': 5, 'margin-left': 5,
                                            'width': 100, 'display': 'inline-block'}), item],
                    id=f"{id}_div", style={'margin-bottom': 10,
                                           'justify': 'center'})


def sidebar_dropdown(label, options, value, id, multi=False, placeholder=None, clearable=True):

    items = [label, dcc.Dropdown(options, value, id=id, multi=multi,
                                 placeholder=placeholder, clearable=clearable,
                                 style={'margin-top': 5})]
    dropdown = html.Div(items, style={'margin-bottom': 10, 'margin-left': 5, 'margin-right': 5})

    return html.Div([dropdown], id=f"{id}_div")


def sidebar_inline_dropdown(label, options, value, id, multi=False, clearable=True):

    text = dcc.Markdown(label, style={"width": 100, 'height': 35,
                                      'margin-top': 5, 'margin-left': 5})
    
    dropdown = dcc.Dropdown(options, value, id=id, multi=multi,
                            placeholder='', clearable=clearable,
                            style={"width": 100, "height": 35,
                                   'display': 'inline-block',
                                   'margin-left': 3})

    return html.Div([text, dropdown], id=f"{id}_div",
                    style={'display': 'flex', 'verticalAlign': 'center', 'height': 35,
                           'margin-left': 0, 'margin-bottom': 10})


def sidebar_range(label, min, max, step, id):

    tooltip = {'always_visible': True,
               'template': '{value}',
               'placement': 'bottom'}
    items = [html.Div(label, style={'margin-bottom': 5}),
             dcc.RangeSlider(min, max, step, marks=None, tooltip=tooltip, id=id)]
    slider = html.Div(items, style={'margin-bottom': 15})

    return html.Div([slider], id=f"{id}_div")


def sidebar_slider(label, min, max, step, value, id):

    dcc.Slider(0, 20, marks=None, value=10)

    items = [html.Div(label, style={'margin-bottom': 5, 'margin-left': 5}),
             dcc.Slider(min, max, step, value=value, marks=None, id=id)]
    slider = html.Div(items, style={'height': 45, 'width': 235, 'margin-bottom': 10})

    return html.Div([slider], id=f"{id}_div")


def sidebar_switch(label, options, id, disabled=False):

    switch_style = {'width': 100,
                    'height': 30,
                    'display': 'inline-block',
                    'margin-left': '3%',
                    'align': 'right'}
    switch_div_style = {'verticalAlign': 'center',
                        'height': 35,
                        'margin-left': 5}

    text = dcc.Markdown(label, style={'display': 'inline-block', "width": 100, 'height': 15,
                                      'text-align': 'justify'})
    options_dict = {"label": options, "value": 1}
    if disabled:
        options_dict['disabled'] = True
    item = dbc.Checklist(options=[options_dict], value=[], id=id, switch=True,
                         style=switch_style)

    return html.Div([text, item], id=f"{id}_div", style=switch_div_style)



def code_box(code_string, id):

    code = dcc.Markdown(code_string, id=f'{id}_code', style={'width': 725})
    clipboard = dcc.Clipboard(id=f'{id}_copy')

    row = [dbc.Col(code, width='auto'), dbc.Col(clipboard, width='auto')]

    code_style = {'border': '2px, black solid',
                  'borderRadius': 10,
                  'padding-top': 10,
                  'background-color': 'white'}

    @callback(Input(f'{id}_copy', 'n_clicks'),
              State(f'{id}_code', 'children'),
              prevent_initial_call=True)
    def update_display(n_clicks, content):

        if n_clicks is not None:
            code_string = content.replace('```python\n', '').replace('\n```', '')
            pyperclip.copy(code_string)

    return html.Div([dcc.Markdown('**Code**:'),
                     dbc.Row(row, align='start', justify='between', style=code_style)])