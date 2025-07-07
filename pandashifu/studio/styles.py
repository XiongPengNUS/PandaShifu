chd_style = 'color:white; background:#007bc2 !important;'

df_styles = [
    {
        "cols": [0],
        "style": {
            "background-color": "#F8F8F8",
            "font-weight": "bold",
            "width": "80px",
        },
    },
    {   
        "style": {
            "max-width": "160px",
            "padding-right": "15px",
        }
    }
]

table_styles =  [
    {'selector': 'th',
     'props': [('background', 'white'),
               ('text-align', 'right'),
               ('font-size', '10pt'),
               ('color', 'black'),
               ('padding-left', '5px'),
               ('padding-right', '5px'),
               ('padding-top', '5px'),
               ('padding-bottom', '5px')]},
    {'selector': 'td',
     'props': [('text-align', 'right'),
               ('font-size', '10pt'),
               ('padding-left', '15px'),
               ('padding-right', '5px'),
               ('padding-top', '5px'),
               ('padding-bottom', '5px')]},
    {'selector': 'tr:nth-of-type(odd)',
     'props': [('background', '#EBEBEB')]},
    {'selector': 'tr:nth-of-type(even)',
     'props': [('background', 'white')]},
]

hc_style = ("font-size:10pt;border:1px solid;"
            "font-family:monospace;"
            "margin-top:6px;padding:3px;padding-left:9px")