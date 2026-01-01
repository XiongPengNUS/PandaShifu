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

icon_button_style = "padding:2px;border:none"

icon_card_style = "background-color:#f8f8f8;margin:0px;padding:0px;box-shadow:none"

icon_card_header_style= "margin:0px;padding:6px;padding-left:12px;font-size:10.5pt"

copy_button_div_style = "display:flex;justify-content:right;height:5px;margin:0px;padding:0px"

code_preview_style = ("font-family:monospace;font-size:10pt;background-color:#f4f4f4;"
                      "padding:10px;border:1px solid #ddd;max-height:120px")