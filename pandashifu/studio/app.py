#app.py - Shiny app file for running the application.

#The file specifies the design of all UI components used in the application.

#Author:  Peng Xiong (xiongpengnus@gmail.com)
#Date:    March 29, 2025

#Version: 1.0
#License: MIT License

import io
import json
from PIL import Image
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#import pyperclip
from xlsxwriter.workbook import Workbook

from shiny import reactive
from shiny.ui import output_plot
from shiny.express import render, ui, input, expressify

from bokeh.models import ColorPicker
from shinywidgets import render_bokeh

from .autosource import *
from .canvas import *
from .styles import *


#current_file = Path(__file__)
#current_directory = current_file.parent

def hr(margin=0.75, offset=0):

    return ui.HTML(f"<hr style='margin-bottom:{margin + offset}em;margin-top:{margin - offset}em'>")


def ui_block(string, btype):

    return ui.markdown(f'<div class="alert alert-block alert-{btype}">\n{string}</div>')


def inline_label(string, pt="5px"):
    
    return ui.HTML(f'<p style="padding-top:{pt}">{string}</p>')


def big_label(string, font=12):

    return ui.HTML(f'<p style="font-size:{font}pt">{string}</p>')


def tool_icon_code(id, figsize):

    return (
        "@render.plot()\n"
        f"def {id}_button_icon():\n"
        "    current_directory = Path(__file__).parent\n"
        f"    fig = plt.figure(figsize={figsize}, facecolor='none')\n"
        f"    img = np.asarray(Image.open(current_directory / 'images/{id}.png'))\n"
        "    plt.imshow(img)\n"
        "    plt.axis('off')\n"
        "    plt.tight_layout(pad=0)\n"
        "    return fig\n"
    )


def tool_effect_code(name, cat):

    id = name.lower().replace(' ', '_')

    return (
        "@reactive.effect\n"
        f"@reactive.event(input.{id}_button)\n"
        f"def to_{id}_section():\n"
        f"    {cat}_selected.set('{name}')\n"
        f"    ui.update_navs('main', selected='{cat}s_panel')\n"
        f"    {cat}_memory.set([])\n"
    )


def tool_disable(disabled):

    for item in ops_menu + dvs_menu + mds_menu:
        item_id = item.lower().replace(' ', '_')
        ui.update_action_button(f"{item_id}_button", disabled=disabled)


def model_variables(data):

    columns = to_column_choices(data.columns)
    col_nums, col_cats, col_nbs = num_cat_labels(data)

    col_predicted = []
    col_predictors = []
    for c in columns:
        if c in col_cats:
            nc = len(data[to_selected_columns(c, data)].unique())
            if  nc > 30:
                continue
            elif nc > 10:
                col_predictors.append(c)
                continue
        col_predicted.append(c)
        col_predictors.append(c)
    
    return col_predicted, col_predictors


def invalid_name(name, error=False):

    try:
        exec(f"{name} = 1")
        if name in var_names.get():
            raise ValueError(f"The variable name '{name}' was already used.")
        return False
    except Exception as err:
        if error:
            return err
        else:
            return True


def display_table(df, min_rows=10):

    if 60 >= df.shape[0]:
        return df

    head = min_rows // 2
    tail = min_rows - head

    df_head = df.head(head)
    df_tail = df.tail(tail)

    columns = df.columns
    ellipsis_row = pd.DataFrame([["..."] * len(columns)],
                                columns=columns, index=["..."])

    return pd.concat([df_head, ellipsis_row, df_tail])

# Global variables and constants
# Default colors for data visuals like bar charts and line plots
default_colors = [c['color'] for c in mpl.rcParams['axes.prop_cycle']]

# Color maps for representing numerical data
num_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

# Color maps for representing categorical data
cat_cmaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
             'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
             'tab20c']

# The menu of all operation tools
ops_menu = ["Value counts operations",
            "Select columns", "Sort rows", "Boolean conditions", "Treat missing values",
            "Correlation", "Aggregation", "Group by", "Pivot table",
            "Add columns", "Time trend", "Clustering", 
            "ANOVA", "Variance inflation factor", "Over sampling"]

# The menu of all data visualization tools
dvs_menu = ["Value counts", "Probability plot", "Histogram", "KDE", "Box plot", "Pair plot",
            "Heat map", "Bar chart", "Radar chart", "Line plot", "Filled areas",
            "Scatter plot", "Regression plot", "ACF and PACF"]

# The menu of all modeling tools
mds_menu = ["Statsmodels", "Sklearn models"]

# Reactive values shared across the app
nodes = reactive.value([])
node_input = reactive.value(dict(data=None, name="data"))
node_highlights = reactive.value([])
arc_highlights = reactive.value([])

canvas_lim = reactive.value([64, -51.2])

op_selected = reactive.value(None)
dv_selected = reactive.value(None)
md_selected = reactive.value(None)

ops = reactive.value(dict(type=None, source=None, data_out=None))
op_memory = reactive.value([])

dvs = reactive.value(dict(type=None, source=None, fig=None, width=640, height=480))
dv_memory = reactive.value([])
init_color = reactive.value(default_colors[0])

mds = reactive.value(dict(type="", source={}, results=None, outputs=None, memory={}))
md_memory = reactive.value({})
md_page = reactive.value(1)


dvs_view = reactive.value(dict(fig=None, width=640, height=480))

model_visual_view = reactive.value(dict(pred=None, reside=None))
node_view = reactive.value(None)

model_outcome = reactive.value("reg")
model_page = reactive.value(1)
model_reset = reactive.value(False)

var_names = reactive.value([])


#ui.page_opts(title="Panda Shifu", full_width=True)

with ui.layout_column_wrap(width="1060px", fixed_width=True):
    with ui.navset_hidden(id="main"):
        with ui.nav_panel(None, value="canvas_panel"):
            ui.HTML('<br>')
            with ui.layout_sidebar(height='900px'):
                with ui.sidebar(width='350px', open="always", bg='#f8f8f8', height="900px"):

                    with ui.navset_tab(id="main_toolset_navs"):
                        button_gap = "10px"
                        #button_heights = "110px"    
                        #icon_size = "140px", "120px"
                        button_heights = "90px"
                        icon_size = "95px", "70px"
                        figsize = (4, 3)

                        with ui.nav_panel("Operations", value="ops_toolset_nav"):
                            with ui.layout_columns(col_widths=(4, 4, 4), gap=button_gap, row_heights=button_heights):
                                tool_ns = globals()
                                for op_name in ops_menu:
                                    op_id = op_name.lower().replace(' ', '_')
                                    exec(tool_icon_code(op_id, figsize), tool_ns)
                                    icon = output_plot(f"{op_id}_button_icon",
                                                    width=icon_size[0], height=icon_size[1])

                                    ui.input_action_button(f"{op_id}_button", "", icon=icon,
                                                        style="padding:0px;padding-top:5px;padding-bottom:5px",
                                                        disabled=True)
                                    exec(tool_effect_code(op_name, "op"), tool_ns)
                                
                        with ui.nav_panel("Visuals", value="dvs_toolset"):
                            with ui.layout_columns(col_widths=(4, 4, 4), gap=button_gap, row_heights=button_heights):
                                tool_ns = globals()
                                for dv_name in dvs_menu:
                                    dv_id = dv_name.lower().replace(' ', '_')
                                    exec(tool_icon_code(dv_id, figsize), tool_ns)
                                    icon = output_plot(f"{dv_id}_button_icon",
                                                    width=icon_size[0], height=icon_size[1])

                                    ui.input_action_button(f"{dv_id}_button", "", icon=icon,
                                                        style="padding:0px;padding-top:5px;padding-bottom:5px",
                                                        disabled=True)
                                    exec(tool_effect_code(dv_name, "dv"), tool_ns)

                        with ui.nav_panel("Models", value="mds_toolset"):
                            with ui.layout_columns(col_widths=(4, 4, 4), gap=button_gap, row_heights=button_heights):
                                tool_ns = globals()
                                for md_name in mds_menu:
                                    md_id = md_name.lower().replace(' ', '_')
                                    exec(tool_icon_code(md_id, figsize), tool_ns)
                                    icon = output_plot(f"{md_id}_button_icon",
                                                    width=icon_size[0], height=icon_size[1])

                                    ui.input_action_button(f"{md_id}_button", "", icon=icon,
                                                        style="padding:0px;padding-top:5px;padding-bottom:5px",
                                                        disabled=True)
                                    exec(tool_effect_code(md_name, "md"), tool_ns)

                with ui.layout_columns(col_widths=(5, 7), gap="20px", height="140px"):
                    
                    with ui.card():
                        ui.card_header("Data file", style=chd_style)
                        ui.input_file("upload_data_file", "",
                                    button_label='Upload', accept=[".csv"], multiple=False, width="100%")

                        @reactive.effect
                        @reactive.event(input.upload_data_file)
                        def load_data_file():
                            file = input.upload_data_file()
                            if file is not None:
                                df = pd.read_csv(file[0]["datapath"])
                                input_dict = node_input.get()
                                input_dict["data"] = df

                                node_list = nodes.get()
                                node_list.clear()
                                view = dict(name="data",
                                            string=df.to_string(max_rows=6, max_cols=6),
                                            shape=df.shape)
                                node_view.set(view)
                                code = f"data = pd.read_csv({file[0]['name'].__repr__()})\ndata"
                                source = dict(name_out="data", code=code, imports=[], markdown="")
                                node_list.append(PSNode((0, 0), "data",
                                                        info=dict(name="data", data=df, view=view, source=source)))
                                node_input.set(dict(name="data", data=df))

                                cs = ["red"]
                                ac = ["gray"]
                                node_highlights.set(cs)
                                arc_highlights.set(ac)

                                var_names.set(["data"])

                                tool_disable(False)
                    
                    with ui.card():
                        ui.card_header("Download", style=chd_style)
                        with ui.layout_columns(col_widths=(4, 4, 4), gap="10px"):
                            @render.download(label="Excel", filename="data.xlsx")
                            def export_data():
                                with io.BytesIO() as buf:
                                    node_list = nodes.get()
                                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                                        workbook = writer.book
                                        fig_index = 1
                                        model_index = 1
                                        for node in node_list:
                                            if node.ntype == "data":
                                                # Output the dataset to the spreadsheet
                                                info = node.info
                                                info["data"].to_excel(writer, sheet_name=info["name"])
                                            elif node.ntype == "model":
                                                # Output the model fitting result to the spreadsheet
                                                view = node.info["view"]
                                                result_cell = pd.DataFrame({'': [view["results"]]})
                                                result_cell.to_excel(writer, sheet_name=f"Model {model_index}")
                                                model_index += 1
                                            elif node.ntype == "visual":
                                                # Output the figure as an image to the spreadsheet
                                                view = node.info["view"]
                                                fig = view["fig"]
                                                
                                                img_buffer = io.BytesIO()
                                                fig.savefig(img_buffer, format="png")
                                                plt.close(fig)

                                                sheet_name = f"Figure {fig_index}"
                                                writer.sheets[sheet_name] = workbook.add_worksheet(sheet_name)
                                                writer.sheets[sheet_name].insert_image("B3", f"plot_{fig_index}.png",
                                                                                    {"image_data": img_buffer})
                                                fig_index += 1
                                    buf.seek(0)
                                    yield buf.getvalue()
                            
                            @render.download(label="Jupyter", filename="code.ipynb")
                            def export_code():
                                node_list = nodes.get()

                                with io.BytesIO() as buf:
                                    notebook_data = export(node_list)
                                    buf.write(json.dumps(notebook_data, indent=4).encode("utf-8"))

                                    buf.seek(0)
                                    yield buf.getvalue()
                            
                            @render.download(label="Python", filename="code.py")
                            def export_py_code():
                                node_list = nodes.get()

                                with io.StringIO() as buf:
                                    notebook_data = export(node_list)
                                    code_list = []
                                    for cell in notebook_data["cells"]:
                                        if cell["cell_type"] == "code":
                                            code_list.append("".join(cell["source"]))
                                        else:
                                            code_list.append("# " + "".join(cell["source"]).replace("\n", "\n# "))
                                    code_string = "\n\n".join(code_list)
                                    
                                    buf.write(code_string)

                                    buf.seek(0)
                                    yield buf.getvalue().encode("utf-8")
                
                with ui.card(height='775px'):
                        @render.express
                        def canvas_plot_func():

                            xmax, ymin = canvas_lim.get()
                            canvas_width, canvas_height = int((xmax + 4) * 12.5), int((3.2 - ymin) * 12.5)
                            output_plot("canvas_plot", click=True)

                            with ui.hold():

                                @render.plot(alt="Canvas plot", width=canvas_width, height=canvas_height)
                                def canvas_plot():
                                    
                                    node_list = nodes.get()
                                    cs = node_highlights.get()
                                    ac = arc_highlights.get()

                                    fig, ax = plt.subplots()
                                    if len(node_list) > 0:
                                        xl, yl = node_list[0].get_all_lines()
                                        ax.plot(xl, yl, color='k', linewidth=2, zorder=0)
                                    for node, c, a in zip(node_list, cs, ac):
                                        pos = node.pos
                                        node_circle = plt.Circle(pos, 1.0,
                                                                facecolor=c, edgecolor='k', linewidth=2,
                                                                zorder=5)
                                        ax.add_patch(node_circle)
                                        anno = "D" if node.ntype == "data" else "V" if node.ntype == "visual" else "M"
                                        ac = "white" if c == "red" else "k"
                                        ax.annotate(anno,  xy=pos, color=ac, fontsize="small", weight="heavy",
                                                    horizontalalignment="center", verticalalignment="center",
                                                    zorder=10)
                                        #if sum(pos) > 0:
                                        if pos[0] > 0:
                                            before = node.before
                                            pos = ((0.35*before.pos[0] + 0.65*node.pos[0]), node.pos[1])
                                            arc_circle = plt.Circle(pos, 0.5,
                                                                    facecolor=a, edgecolor='k',
                                                                    linewidth=2, zorder=1)
                                            ax.add_patch(arc_circle)

                                    ax.set_xlim([-4, xmax])
                                    ax.set_ylim([ymin, 3.2])
                                    ax.axis('off')

                                    return fig
                                
                        events = (input.canvas_plot_click, input.upload_data_file, input.cancel_data_button)
                        @reactive.effect
                        @reactive.event(*events)
                        def update_canvas_plot():
                            clk = input.canvas_plot_click()
                            cs = node_highlights.get()
                            ac = arc_highlights.get()
                            selected_idx = None if 'red' not in cs else cs.index('red')
                            node_list = nodes.get()
                            xn = [n.pos[0] for n in node_list]
                            yn = [n.pos[1] for n in node_list]

                            input_dict = dict(name="", data=None)
                            if clk is not None:
                                cs = ['gray'] * len(cs)
                                ac = ['gray'] * len(cs)
                                for i, (x, y) in enumerate(zip(xn, yn)):
                                    if (clk['x'] - x)**2 + (clk['y'] - y)**2 < 1:
                                        if i != selected_idx:
                                            cs[i] = 'red'
                                            node = node_list[i]
                                            if node.ntype == "data":
                                                node_info = node.info
                                                input_dict = dict(name=node_info["name"],
                                                                data=node_info["data"])
                                                node_input.set(input_dict)
                                                node_view.set(node_list[i].info["view"])
                                            elif node.ntype == "visual":
                                                node_view.set(node_list[i].info["view"])
                                            elif node.ntype == "model":
                                                node_view.set(node_list[i].info["view"])
                                            
                                        break

                                    if (clk['x'] - x + 3.5)**2 + (clk['y'] - y)**2 < 0.5**2:
                                        node = node_list[i]
                                        if i > 0 and ac[i] != 'orange':
                                            node_view.set(node.info["source"])
                                            ac[i] = "orange"
                                            break
                                        
                                node_highlights.set(cs)
                                arc_highlights.set(ac)
                                disabled = input_dict["data"] is None

                                tool_disable(disabled)

            @render.express
            def float_node_view():
                view = node_view.get()
                pos = dict(left="35%", bottom="10%")
                if view is not None:
                    if "string" in view:
                        with ui.panel_absolute(draggable=True, width="590px", **pos):
                            with ui.panel_well():
                                ui.card_header("Dataset", style=chd_style)
                                with ui.card():#(height=270):
                                    row, col = view["shape"]
                                    #ui.markdown(f"```{view['name']}```: {row} rows x {col} columns\n"
                                    #            f"```\n{view['string']}\n```")
                                    ui.markdown(
                                        f"<pre style='font-size:12px'><code>{view['name']}:</code>"
                                        f" {row} rows x {col} columns<br><br>"
                                        f"<code>{view['string'].replace('\n', '<br>')}</code></pre>"
                                    )
                                ui.input_action_button("close_data_view", "Close", width="110px")

                    elif "width" in view and "height" in view:
                        fig = view["fig"]
                        fig.set_dpi(60)
                        width = int(np.minimum(view["width"]*3/5, 600))
                        height = view["height"]*3/5
                        with ui.panel_absolute(draggable=True, width=f"{width + 90}px", **pos):
                            with ui.panel_well():
                                ui.card_header("Figure", style=chd_style)
                                with ui.card(height=height+20, max_height=480):
                                    @render.plot(width=width, height=height)
                                    def fig_view_plot():
                                        return fig
                                ui.input_action_button("close_fig_view", "Close", width="110px")
                    elif "code" in view:
                        with ui.panel_absolute(draggable=True, width=f"550px", **pos):
                            with ui.panel_well():
                                ui.card_header("Source", style=chd_style)
                                code = view["code"]
                                if isinstance(code, dict):
                                    keys = ["vars", "dummy", "pipeline", "fitting"]
                                    code = '\n'.join([code[k] for k in keys])
                                markdown = view["markdown"]
                                clines = code.split("\n")
                                if len(clines) > 15:
                                    clines = clines[:6] + ["... ..."] * 3 + clines[-6:]
                                    code = "\n".join(clines)
                                with ui.card():
                                    ui.markdown(view["markdown"])
                                    hr(0.5)
                                    #with ui.layout_columns(col_widths=(-10, 2)):
                                    #    ui.input_action_link("node_view_copy", "copy", style="height:1px")
                                    #@render.code
                                    #def source_code_view():
                                    #    return code
                                    code_html = code.replace("\n", "<br>")
                                    ui.markdown(f"<pre style='font-size:12px'><code>{code_html}</code></pre>")

                                ui.input_action_button("close_source_view", "Close", width="110px")
                    elif "results" in view:
                        with ui.panel_absolute(draggable=True, width=f"650px", **pos):
                            with ui.panel_well():
                                ui.card_header("Model", style=chd_style)
                                results = view["results"]
                                with ui.card(max_height=280):
                                    #@render.code
                                    #def source_code_view():
                                    #    return results
                                    results_html = results.replace("\n", "<br>")
                                    ui.markdown(f"<pre style='font-size:12px'><code>{results_html}</code></pre>")
                                ui.input_action_button("close_model_view", "Close", width="110px")

            @reactive.effect
            @reactive.event(input.close_data_view)
            def close_data_view_button():
                node_view.set(None)
            
            @reactive.effect
            @reactive.event(input.close_fig_view)
            def close_fig_view_button():
                node_view.set(None)
            
            @reactive.effect
            @reactive.event(input.close_model_view)
            def close_model_view_button():
                node_view.set(None)

            @reactive.effect
            @reactive.event(input.close_source_view)
            def close_source_view_button():
                node_view.set(None)

        with ui.nav_panel(None, value="ops_panel"):
            ui.HTML('<br>')
            with ui.layout_sidebar(height='900px'):
                with ui.sidebar(bg='#f8f8f8', width='350px', height='900px'):
                    
                    @render.express
                    def ops_panel_ui():
                        
                        node = node_input.get()
                        data_in = node["data"]
                        name_in = node["name"]
                        ops_dict = ops.get()
                        if data_in is None:
                            return

                        columns = to_column_choices(data_in.columns)
                        col_nums, col_cats, col_nbs = num_cat_labels(data_in)
                        aggs = ['count', 'mean', 'median', 'std', 'var', 'min', 'max', 'sum']
                        aggs_default = ['count', 'mean']

                        op_type = op_selected.get()
                        ui.markdown(f"#### {op_type}")

                        if op_type == "Value counts operations":
                            count_choices = columns
                            ui.input_selectize("counts_ops_selectize", "Columns",
                                            choices=count_choices, selected=[],
                                            multiple=True)
                            
                            @render.express(inline=True)
                            def counts_ops_unstack_ui():
                                selected = list(input.counts_ops_selectize())
                                maxItems = len(selected) - 1 if len(selected) > 1 else 0
                                ui.input_selectize("counts_ops_unstack_selectize", "Unstack levels",
                                                choices=selected, selected=[],
                                                multiple=True, remove_button=True,
                                                options={"placeholder": "None", "maxItems": maxItems})

                            @render.express(inline=True)
                            def counts_ops_sort_by_ui():
                                unstack = list(input.counts_ops_unstack_selectize())
                                if len(unstack) == 0:
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_switch("counts_ops_sort_switch",
                                                        "Sort", value=True)
                                        ui.input_switch("counts_ops_sort_descending_switch",
                                                        "Descending", value=True)
                            
                            with ui.layout_columns(col_widths=(6, 6)):
                                ui.input_switch("counts_ops_normalize_switch", "Normalize")
                                ui.input_switch("counts_ops_reset_switch", "Reset index")
        
                        elif op_type == "Select columns":
                            ui.input_selectize("select_columns_selectize", "Columns",
                                            choices=columns, selected=columns,
                                            multiple=True)
                        elif op_type == "Sort rows":
                            ui.input_selectize("sort_columns_selectize", "Sort on columns",
                                            choices=columns, selected=[], multiple=True,
                                            remove_button=True)
                            with ui.layout_columns(col_widths=(6, 6)):
                                ui.input_switch("sort_descending_switch", "Descending")
                                ui.input_switch("sort_reset_switch", "Reset index")
                        elif op_type == "Boolean conditions":
                            with ui.layout_columns(col_widths=(8, 4)):
                                ui.input_selectize("filter_column_selectize", "Column on the left",
                                                choices=[""] + columns)
                                filter_operators = ["", "==", "!=", "<=", "<", ">=", ">", "in", "not in"]
                                ui.input_selectize("filter_operator_selectize", "Operator",
                                                choices=filter_operators, selected="")
                            ui.input_text("filter_value_text", "Value on the right")

                            with ui.layout_columns(col_widths=(-5, 7)):
                                ui.input_action_button("add_filter_button", "New condition")
                            
                            @reactive.effect
                            @reactive.event(input.filter_column_selectize,
                                            input.filter_operator_selectize,
                                            input.filter_value_text)
                            def add_filter_button_disable():
                                cond1 = input.filter_column_selectize() == ""
                                cond2 = input.filter_operator_selectize() == ""
                                cond3 = str_to_values(input.filter_value_text(), sup=True) is None
                                ui.update_action_button("add_filter_button", disabled=(cond1 or cond2 or cond3))
                            
                            with ui.layout_columns(col_widths=(6, 6)):
                                ui.input_switch("filter_select_rows_switch", "Filter rows")

                                @render.express
                                def filter_reset_index_ui():
                                    if input.filter_select_rows_switch():
                                        ui.input_switch("filter_reset_switch", "Reset index")
                            
                            @render.express
                            def filter_condition_column_ui():
                                if not input.filter_select_rows_switch():
                                    ui.input_text("filter_condition_column_text", "To column")

                        elif op_type == "Correlation":
                            ui.input_selectize("corr_metric_selectize", "Metric",
                                            choices=["Correlation", "Covariance"])
                            ui.input_selectize("corr_columns_selectize", "Columns",
                                            choices=[""] + col_nbs, selected=col_nbs, multiple=True)
                            ui.input_selectize("corr_drops_selectize", "Drop rows", choices=[],
                                            multiple=True, remove_button=True,
                                            options={"placeholder": "None"})
                            
                            @reactive.effect
                            @reactive.event(input.corr_columns_selectize)
                            def orr_drops_selectize_update_choices():
                                cols = input.corr_columns_selectize()
                                if len(cols) > 0:
                                    ui.update_selectize("corr_drops_selectize", choices=cols, selected=[])
                                else:
                                    ui.update_selectize("corr_drops_selectize", choices=[])
                        elif op_type == "Aggregation":
                            ui.input_selectize('agg_columns_selectize', 'Columns',
                                            choices=[""] + col_nbs, selected="", multiple=True)
                            ui.input_selectize("agg_methods_selectize", 'Methods',
                                            choices=aggs, selected=aggs_default, multiple=True)
                            ui.input_switch("agg_transpose_switch", "Transpose")
                        elif op_type == "Group by":
                            ui.input_selectize("group_by_columns_selectize", "Group by",
                                            choices=[""] + columns,
                                            multiple=True, remove_button=True)
                            ui.input_selectize("group_view_columns_selectize", "View on",
                                            choices=[""] + columns,
                                            multiple=True, remove_button=True)
                            ui.input_selectize("group_methods_selectize", "Methods",
                                            choices=[""] + aggs,
                                            multiple=True, remove_button=True)
                            with ui.layout_columns(col_widths=(6, 6)):
                                ui.input_switch("group_reset_switch", "Reset index")
                                ui.input_switch("group_transpose_switch", "Transpose")
                        elif op_type == "Pivot table":
                            ui.input_selectize("pivot_values_selectize", "View on",
                                            choices=[""] + columns, multiple=True, remove_button=True)
                            ui.input_selectize("pivot_index_selectize", "Row index",
                                            choices=[""] + columns, multiple=True, remove_button=True)
                            ui.input_selectize("pivot_columns_selectize", "Columns", choices=[""] + col_cats,
                                            multiple=True, remove_button=True)
                            ui.input_selectize("pivot_methods_selectize", "Methods", 
                                            choices=[""] + aggs, multiple=True, remove_button=True)
                            with ui.layout_columns(col_widths=(6, 6)):
                                ui.input_switch("pivot_reset_switch", "Reset index")
                                ui.input_switch("pivot_transpose_switch", "Transpose")
                        elif op_type == "Treat missing values":
                            nan_columns = to_column_choices(data_in.columns[data_in.isnull().sum() > 0])
                            ui.input_selectize("nan_columns_selectize", "Columns", choices=[""]+nan_columns,
                                            multiple=True, remove_button=True,
                                            options={"placeholder": "All columns"})
                            ui.input_selectize("nan_method_selectize", "Method", choices=["drop", "fill"])
                            @render.express
                            def nan_conditional_ui():
                                if input.nan_method_selectize() == "fill":
                                    ui.input_text("nan_fill_value_text", "Value to fill")
                                elif input.nan_method_selectize() == "drop":
                                    ui.input_switch("nan_reset_switch", "Reset index")
                        elif op_type == "Time trend":
                            ui.input_selectize("time_trend_columns_selectize", "Columns",
                                            choices=[""] + col_nums,
                                            multiple=True, remove_button=True)
                            with ui.layout_columns(col_widths=(4, 8)):
                                transforms =  ["change", "relative change", "log change", "moving average"]
                                inline_label("Transform")
                                ui.input_selectize("time_trend_transform_selectize", "",
                                                choices=[""] + transforms)
                                inline_label("Steps")
                                ui.input_text("time_trend_steps_text", "", placeholder="1")
                            ui.input_switch("time_trend_drop_original_data", "Drop original data")
                        elif op_type == "ANOVA":
                            ui.input_selectize("anova_target_selectize", "Numerical target",
                                            choices=[""] + col_nums)
                            ui.input_selectize("anova_features_selectize", "Features",
                                            choices=[""], multiple=True, remove_button=True)
                            ui.input_text("anova_formula_text", "Formula")

                            with ui.layout_columns(col_widths=(2, 4, 2, 4)):
                                inline_label("Type")
                                ui.input_selectize("anova_type_selectize", "", choices=["I", "II", "III"])
                                inline_label("Test")
                                ui.input_selectize("anova_test_selectize", "", choices=["F", "Chisq", "Cp"])

                            @reactive.effect
                            @reactive.event(input.anova_target_selectize, ignore_init=True)
                            def anoava_features_choices_update():
                                node = node_input.get()
                                data_in = node["data"]
                                max_cats = np.minimum(len(data_in)//10, 100)
                                target = to_selected_columns(input.anova_target_selectize(), data_in)
                                if target != "":
                                    xdata = data_in.drop(columns=[target])
                                    feature_choices = discrete_labels(xdata, max_cats=max_cats)
                                    ui.update_selectize("anova_features_selectize",
                                                        choices=[""] + feature_choices, selected="")
                                else:
                                    ui.update_selectize("anova_features_selectize",
                                                        choices=[""] + columns)
                            
                            @reactive.effect
                            @reactive.event(input.anova_target_selectize,
                                            input.anova_features_selectize, ignore_init=True)
                            def anova_formula_text_update():
                                target = to_selected_columns(input.anova_target_selectize(), data_in)
                                features = to_selected_columns(input.anova_features_selectize(), data_in)
                                if target != "" and len(features) > 0:
                                    features_list = [f"C({item})" if item in col_nbs else item for item in features]
                                    formula = f"{target} ~ {' + '.join(features_list)}"
                                    ui.update_text("anova_formula_text", value=formula)
                                else:
                                    ui.update_text("anova_formula_text", value="")

                        elif op_type == "Variance inflation factor":
                            #_, feature_choices = model_variables(data_in)
                            feature_choices = [col for col in col_nbs if
                                            data_in[col].isnull().sum() == 0 and
                                            data_in[col].min() < data_in[col].max()]
                            ui.input_selectize("vif_features_selectize", "Features",
                                            choices=feature_choices, selected=col_nums,
                                            multiple=True)
                            with ui.layout_columns(col_widths=(6, 6)):
                                ui.input_switch("vif_add_constant_switch", "Intercept", value=True)
                                ui.input_switch("vif_reset_switch", "Reset index", value=True)

                        elif op_type == "Clustering":
                            ui.input_selectize("clustering_method_selectize", "Method",
                                            choices=["K-means clustering", "Hierarchical clustering"])
                            ui.input_selectize("clustering_columns_selectize", "Features for clustering",
                                            choices=[""] + col_nbs, multiple=True, remove_button=True)
                            ui.input_text("clustering_numbers_text", "Numbers of clusters")
                        elif op_type == "Over sampling":
                            ui.input_selectize("over_sampling_target_selectize", "Categorical target",
                                            choices=[""] + col_cats)
                            ui.input_selectize("over_sampling_features_selectize", "Features",
                                            choices=[""] + columns,
                                            multiple=True, remove_button=True)
                            ui.input_selectize("over_sampling_method_selectize", "Method",
                                            choices=["", "Random over-sampling", "SMOTE", "ADASYN"])
                            with ui.layout_columns(col_widths=(6, 6)):
                                strategies = ["auto", "minority", "not minority", "not majority", "all"]
                                ui.input_selectize("over_sampling_strategy_selectize", "Strategy",
                                                choices=strategies)
                                ui.input_numeric("over_sampling_k_neighbors_numeric", "Neighbor No.",
                                                min=1, max=50, step=1, value=5)
                            
                            @reactive.effect
                            @reactive.event(input.over_sampling_target_selectize)
                            def over_sampling_features_choices_update():
                                target = input.over_sampling_target_selectize()
                                if target != "":
                                    cols = [c for c in columns if c != target]
                                    ui.update_selectize("over_sampling_features_selectize",
                                                        choices=cols, selected=cols)
                                else:
                                    ui.update_selectize("over_sampling_features_selectize",
                                                        choices=[""] + columns, selected="")
                            
                            @reactive.effect
                            @reactive.event(input.over_sampling_features_selectize)
                            def over_sampling_method_ui_update():
                                node = node_input.get()
                                data_in = node["data"]
                                features = to_selected_columns(input.over_sampling_features_selectize(), data_in)
                                xdata = data_in[features]
                                is_num = xdata.apply(is_numeric_dtype, axis=0).values
                                if all(is_num):
                                    choices = ["", "Random over-sampling", "SMOTE", "ADASYN"]
                                else:
                                    choices = ["", "Random over-sampling", "SMOTE"]
                                ui.update_selectize("over_sampling_method_selectize",
                                                    choices=choices, selected="")

                        elif op_type == "Add columns":
                            choices = ["Arithmetic expression",
                                       "Type conversion", "String operations",
                                       "To date time", "To dummies", "To segments"]
                            ui.input_selectize("add_cols_type_selectize", "Expression type",
                                               choices=choices)
                            
                            label_dict = {"Arithmetic expression": "Formula",
                                          "Type conversion": "Data type",
                                          "String operations": "Methods",
                                          "To date time": "Format",
                                          "To dummies": "",
                                          "To segments": "Bins",
                                          "": "Formula"}

                            @render.express(inline=True)
                            def add_cols_from():

                                exp_type = input.add_cols_type_selectize()
                                if exp_type in choices[:2]:
                                    cols = columns
                                elif exp_type in choices[2:-1]:
                                    cols = col_cats
                                else:
                                    cols = col_nums
                                
                                multiple = exp_type == choices[0]
                                
                                ui.input_selectize("add_cols_from_columns_selectize", "From column(s)",
                                                choices=[""] + cols, 
                                                multiple=multiple, remove_button=multiple)
                                ui.input_text("add_cols_to_columns_text", "To column(s)")

                                if exp_type == "To dummies":
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_switch("add_cols_drop_switch", "Drop first")
                                        ui.input_switch("add_cols_binary_switch", "To binaries")
                                else:
                                    label = label_dict[exp_type]
                                    ui.input_text("add_cols_expression_text", label)
                                    if exp_type == "To segments":
                                        ui.input_text("add_cols_labels_text", "Labels", placeholder="None")        

                            @reactive.effect
                            @reactive.event(input.add_cols_type_selectize)
                            def reset_add_cols():

                                ui.update_selectize("add_cols_columns_selectize", selected="")
                            
                            @reactive.effect
                            @reactive.event(input.add_cols_from_columns_selectize)
                            def update_formula():

                                from_cols = to_selected_columns(input.add_cols_from_columns_selectize(), data_in)
                                if input.add_cols_type_selectize() == "Arithmetic expression":
                                    terms = [f"{name_in}[{c.__repr__()}]" for c in from_cols]
                                    ui.update_text("add_cols_expression_text", value=" + ".join(terms))
                        
                        hr(0)
                        ui.input_text("op_name_out_text", "Output name",
                                    placeholder="Key in a variable name...")
                        
                        ui.input_text_area("op_markdown_text_area", "Markdown",
                                        placeholder="Key in notes...", height="100px")
                        
                        #ui.input_action_button("op_debug", "Debug")
                        #@render.code
                        #@reactive.event(input.op_debug)
                        #def op_debug_display():
                        #    return str(ops_dict)

                    @reactive.effect
                    @reactive.event(input.add_filter_button, ignore_init=True)
                    def add_filter_button_action():
                        
                        filters = op_memory.get()
                        filters.append(dict(column=input.filter_column_selectize(),
                                            operator=input.filter_operator_selectize(),
                                            value=input.filter_value_text()))
                        ui.update_selectize("filter_column_selectize", selected="")
                        ui.update_selectize("filter_operator_selectize", selected="")
                        ui.update_text("filter_value_text", value="")

                ui.input_switch("op_show_code_switch", "Show code")

                @render.express(inline=True)
                def op_source_results_ui():

                    node = node_input.get()
                    data_in = node["data"]
                    name_in = node["name"]
                
                    ops_dict = ops.get()
                    source = ops_dict["source"]

                    name_out = input.op_name_out_text().strip()
                    source = operation_source(op_selected.get(), name_in, data_in, input, op_memory.get())
                    ops_dict["type"] = op_selected.get()
                    ops_dict["source"] = source

                    if name_out in var_names.get():
                        msg = f"The variable name '{name_out}' was already used."
                        source["error"] = msg
                        data_out = msg
                    else:
                        source["error"] = None
                        data_out = operation_exec_source(data_in, name_in, source)
                    ops_dict["data_out"] = data_out

                    if input.op_show_code_switch():
                        
                        @render.ui
                        @reactive.event(input.op_markdown_text_area)
                        def op_markdown_display():
                            if input.op_markdown_text_area().strip() != "":
                                return ui.markdown(input.op_markdown_text_area())
                                    
                        @render.code
                        def op_code_display():
                            imports = source["imports"]
                            imports_code = f"{'\n'.join(imports)}\n\n" if len(imports) > 0 else ""
                            return (
                                f"{imports_code}"
                                f"{source['code']}"
                            )

                        hr()
                
                #with ui.layout_columns(col_widths=(3, -9)):
                #    with ui.layout_columns(col_widths=(6, 6)):
                #        inline_label("Rows to display")
                #        ui.input_selectize("data_min_rows", "",
                #                           choices=["10", "16", "30", "50", "100", "1000"])

                    with ui.card(height="720px", full_screen=True):

                        if isinstance(data_out, str):
                            ui_block(f"<b>Error</b>: {data_out}", 'danger')
                        else:
                            row, col = data_out.shape
                            table_width = len(data_out.__repr__().split('\n')[0]) * 72 // 96
                            
                            with ui.layout_column_wrap(width=f"{table_width}px",
                                                       fixed_width=True, fill=False, fillable=False):
                                @render.table()
                                def data_preview():
                                    table = display_table(data_out, 16).style.format(precision=4)
                                    table.set_caption(f"{row} rows x {col} columns")
                                    return table.set_table_styles(table_styles)

                    @reactive.effect
                    def operation_save_update_disable():
                        
                        ops_dict = ops.get()
                        if ops_dict["source"] is None:
                            ui.update_action_button('save_data', disabled=True)
                        else:
                            name_out = ops_dict["source"]["name_out"]
                            data_out = ops_dict["data_out"]
                            if isinstance(data_out, str) or data_out is None or name_out == "":
                                ui.update_action_button('save_data_button', disabled=True)
                            else:
                                ui.update_action_button('save_data_button', disabled=False)

                with ui.layout_columns(col_widths=(2, -8, 2)):
                    ui.input_action_button("cancel_data_button", "Cancel", value=0)
                    ui.input_action_button("save_data_button", "Save")

                @reactive.effect
                @reactive.event(input.cancel_data_button, input.save_data_button)
                def save_cancel_data_button_action():

                    ui.update_text("op_name_out_text", value="")
                    ui.update_text_area("op_markdown_text_area", value="")
                    op_memory.set([])

                @reactive.effect
                @reactive.event(input.save_data_button)
                def save_data_button_action():

                    node_list = nodes.get()
                    ops_dict = ops.get()
                    source = ops_dict["source"]
                    name_out = source["name_out"]
                    data_out = ops_dict["data_out"]
                    
                    view = dict(name=name_out,
                                string=data_out.to_string(max_rows=6, max_cols=6),
                                shape=data_out.shape)
                    node_view.set(view)
                    
                    cs = node_highlights.get()
                    root = node_list[cs.index("red")]
                    info = dict(name=name_out, data=data_out,
                                view=view, source=source)
                    node_list.append(root.grow("data", info=info))
                    node_input.set(dict(name=name_out, data=data_out, view=view))

                    cs = ['gray'] * (len(node_list) - 1) + ['red']
                    node_highlights.set(cs)
                    arc_highlights.set(["gray"] * len(cs))

                    all_names = var_names.get()
                    all_names.append(name_out)

        with ui.nav_panel(None, value="dvs_panel"):
            color = reactive.value('#1f77b4')
            with ui.layout_sidebar(height="900px"):
                with ui.sidebar(bg='#f8f8f8', width='350px', height='900px'):
                
                    @render.express
                    def dvs_panel_ui():
                        dv_type = dv_selected.get()
                        ui.markdown(f"#### {dv_type}")

                        node = node_input.get()
                        data = node["data"]
                        if data is None:
                            return
                        
                        dvs_dict = dvs.get()

                        with ui.navset_tab(id="visual_config_nav"):
                            with ui.nav_panel("Plot"):

                                columns = to_column_choices(data.columns)
                                col_nums, col_cats, col_nbs = num_cat_labels(data)

                                if dv_type == "Value counts":
                                    choices = [""] + discrete_labels(data, max_cats=100)
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Column")
                                        ui.input_selectize("value_counts_column_selectize", "", choices=choices)

                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_selectize("value_counts_direction_selectize", "Direction",
                                                           choices=["Vertical", "Horizontal"])
                                        ui.input_selectize("value_counts_method_selectize", "Method",
                                                           choices=["Count", "Density"])
                                    
                                    @reactive.effect
                                    @reactive.event(input.value_counts_column_selectize,
                                                    input.value_counts_direction_selectize,
                                                    input.value_counts_method_selectize)
                                    def value_counts_labels_update():

                                        column = input.value_counts_column_selectize()
                                        direction =  input.value_counts_direction_selectize()
                                        method = input.value_counts_method_selectize()

                                        if direction == "Vertical":
                                            ui.update_text("fig_xlabel_text", value=column)
                                            ui.update_text("fig_ylabel_text", value=method)
                                        elif direction == "Horizontal":
                                            ui.update_text("fig_xlabel_text", value=method)
                                            ui.update_text("fig_ylabel_text", value=column)

                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Palette", pt="8px")
                                        with ui.layout_columns(col_widths=(6, 6), gap="2px"):

                                            @render.ui
                                            def value_counts_hexcolor():
                                                c = color.get()
                                                return ui.HTML(f"<span style='{hc_style}'>{c}</span>")

                                            @render_bokeh(height="36px", width="100%", fill=True)
                                            def value_counts_color_picker():
                                                picker = ColorPicker(
                                                    title="",
                                                    color='#1f77b4',
                                                    min_width=95
                                                )

                                                def update_value_counts_picker_color(attr, old, new):
                                                    color.set(new)

                                                picker.on_change('color', update_value_counts_picker_color)
                                                return picker
                                    
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("value_counts_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)
                                
                                elif dv_type == "Histogram":
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Column")
                                        ui.input_selectize("hist_column_selectize", "",
                                                           choices=[""]+col_nums)

                                        choices = [""] + discrete_labels(data, max_cats=8)
                                        inline_label("Group")
                                        ui.input_selectize("hist_group_by_selectize", "",
                                                           choices=choices, remove_button=True,
                                                           options={"placeholder": "None"})
                                    
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_numeric("hist_bins_numeric", "Bins", min=5, max=80, value=10)
                                        ui.input_selectize("hist_method_selectize", "Method",
                                                           choices=["Count", "Density"])
                                    
                                    @reactive.effect
                                    @reactive.event(input.hist_column_selectize,
                                                    input.hist_method_selectize)
                                    def hist_labels_update():

                                        column = input.hist_column_selectize()
                                        method = input.hist_method_selectize()

                                        ui.update_text("fig_xlabel_text", value=column)
                                        ui.update_text("fig_ylabel_text", value=method)
                                        
                                    with ui.navset_hidden(id="hist_conditional_ui"):
                                        with ui.nav_panel(None, value="hist_single_case"):
                                            with ui.layout_columns(col_widths=(3, 9)):
                                                inline_label("Palette", pt="8px")
                                                with ui.layout_columns(col_widths=(6, 6), gap="2px"):

                                                    @render.ui
                                                    def hist_hexcolor():
                                                        c = color.get()
                                                        return ui.HTML(f"<span style='{hc_style}'>{c}</span>")

                                                    @render_bokeh(height="36px", width="100%", fill=True)
                                                    def hist_color_picker():
                                                        picker = ColorPicker(
                                                            title="",
                                                            color='#1f77b4',
                                                            min_width=95
                                                        )
                                            
                                                        def update_hist_picker_color(attr, old, new):
                                                            color.set(new)

                                                        picker.on_change('color', update_hist_picker_color)
                                                        return picker
                                        
                                        with ui.nav_panel(None, value="hist_multiple_case"):
                                            with ui.layout_columns(col_widths=(6, 6)):
                                                ui.input_selectize("hist_grouped_norm_selectize", "Normalized",
                                                                choices=["Separately", "Jointly"])
                                                ui.input_selectize("hist_grouped_multiple_selectize", "Style",
                                                                choices=["Layer", "Stack", "Fill"])
                                            with ui.layout_columns(col_widths=(3, 9)):
                                                inline_label("Colormap")
                                                ui.input_selectize("hist_grouped_cmap_selectize", "",
                                                                choices=cat_cmaps, selected="tab10")
                                    
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("hist_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)
                                    
                                    @reactive.effect
                                    @reactive.event(input.hist_group_by_selectize)
                                    def hist_group_by_selectize_update_ui():

                                        if input.hist_group_by_selectize() == "":
                                            ui.update_navs("hist_conditional_ui", selected="hist_single_case")
                                        else:
                                            ui.update_navs("hist_conditional_ui", selected="hist_multiple_case")
                                
                                elif dv_type == "KDE":
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Column")
                                        ui.input_selectize("kde_column_selectize", "",
                                                        choices=[""]+col_nums)

                                        choices = [""] + discrete_labels(data, max_cats=8)
                                        inline_label("Group")
                                        ui.input_selectize("kde_group_by_selectize", "",
                                                        choices=choices, remove_button=True,
                                                        options={"placeholder": "None"})
                                    
                                    @reactive.effect
                                    @reactive.event(input.kde_column_selectize)
                                    def hist_labels_update():

                                        column = input.kde_column_selectize()
                                        
                                        ui.update_text("fig_xlabel_text", value=column)
                                        ui.update_text("fig_ylabel_text", value="Density")
                                        
                                    with ui.navset_hidden(id="kde_conditional_ui"):
                                        with ui.nav_panel(None, value="kde_single_case"):
                                            with ui.layout_columns(col_widths=(3, 9)):
                                                inline_label("Palette", pt="8px")
                                                with ui.layout_columns(col_widths=(6, 6), gap="2px"):

                                                    @render.ui
                                                    def kde_hexcolor():
                                                        c = color.get()
                                                        return ui.HTML(f"<span style='{hc_style}'>{c}</span>")

                                                    @render_bokeh(height="36px", width="100%", fill=True)
                                                    def kde_color_picker():
                                                        picker = ColorPicker(
                                                            title="",
                                                            color='#1f77b4',
                                                            min_width=95
                                                        )
                                            
                                                        def update_kde_picker_color(attr, old, new):
                                                            color.set(new)

                                                        picker.on_change('color', update_kde_picker_color)
                                                        return picker
                                        
                                        with ui.nav_panel(None, value="kde_multiple_case"):
                                            with ui.layout_columns(col_widths=(6, 6)):
                                                ui.input_selectize("kde_grouped_norm_selectize", "Normalized",
                                                                choices=["Separately", "Jointly"])
                                                ui.input_selectize("kde_grouped_multiple_selectize", "Style",
                                                                choices=["Layer", "Stack", "Fill"])
                                            with ui.layout_columns(col_widths=(3, 9)):
                                                inline_label("Colormap")
                                                ui.input_selectize("kde_grouped_cmap", "",
                                                                choices=cat_cmaps, selected="tab10")
                                    
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("kde_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)
                                    
                                    @reactive.effect
                                    @reactive.event(input.kde_group_by_selectize)
                                    def kde_group_by_selectize_update_ui():

                                        if input.kde_group_by_selectize() == "":
                                            ui.update_navs("kde_conditional_ui", selected="kde_single_case")
                                        else:
                                            ui.update_navs("kde_conditional_ui", selected="kde_multiple_case")

                                elif dv_type == "Box plot":
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Column")
                                        ui.input_selectize("boxplot_column_selectize", "",
                                                           choices=[""]+col_nums)

                                        groups = [""] + discrete_labels(data, max_cats=50)
                                        inline_label("Group")
                                        ui.input_selectize("boxplot_group_by_selectize", "",
                                                           choices=groups, remove_button=True,
                                                           options={"placeholder": "None"})
                                    
                                        hues = [""] + discrete_labels(data, max_cats=8)
                                        inline_label("Hues")
                                        ui.input_selectize("boxplot_hue_selectize", "",
                                                           choices=hues, remove_button=True,
                                                           options={"placeholder": "None"})

                                    @reactive.effect
                                    @reactive.event(input.boxplot_column_selectize,
                                                    input.boxplot_group_by_selectize,
                                                    input.boxplot_direction_selectize)
                                    def boxplot_labels_update():

                                        column = input.boxplot_column_selectize()
                                        group = input.boxplot_group_by_selectize()
                                        if input.boxplot_direction_selectize() == "Horizontal":
                                            group, column = column, group
                                        ui.update_text("fig_xlabel_text", value=group)
                                        ui.update_text("fig_ylabel_text", value=column)
                                    
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_switch("boxplot_notch_switch", "Notch")
                                        ui.input_switch("boxplot_mean_switch", "Mean")
                                        ui.input_selectize("boxplot_direction_selectize", "Direction",
                                                           choices=["Vertical", "Horizontal"])
                                        ui.input_numeric("boxplot_width_numeric", "Width",
                                                         min=0.1, max=1, step=0.05, value=0.8)
                                    
                                    with ui.navset_hidden(id="boxplot_conditional_ui"):
                                        with ui.nav_panel(None, value="boxplot_single_case"):
                                            with ui.layout_columns(col_widths=(3, 9)):
                                                inline_label("Palette", pt="8px")
                                                with ui.layout_columns(col_widths=(6, 6), gap="2px"):

                                                    @render.ui
                                                    def boxplot_hexcolor():
                                                        c = color.get()
                                                        return ui.HTML(f"<span style='{hc_style}'>{c}</span>")

                                                    @render_bokeh(height="36px", width="100%", fill=True)
                                                    def boxplot_color_picker():
                                                        picker = ColorPicker(
                                                            title="",
                                                            color='#1f77b4',
                                                            min_width=95
                                                        )
                                            
                                                        def update_boxplot_picker_color(attr, old, new):
                                                            color.set(new)

                                                        picker.on_change('color', update_boxplot_picker_color)
                                                        return picker
                                        
                                        with ui.nav_panel(None, value="boxplot_multiple_case"):
                                            with ui.layout_columns(col_widths=(3, 9)):
                                                inline_label("Colormap")
                                                ui.input_selectize("boxplot_grouped_cmap_selectize", "",
                                                                   choices=cat_cmaps, selected="tab10")
                                        
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("boxplot_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)

                                    @reactive.effect
                                    @reactive.event(input.boxplot_hue_selectize)
                                    def boxplot_group_by_selectize_update_ui():

                                        if input.boxplot_hue_selectize() == "":
                                            ui.update_navs("boxplot_conditional_ui",
                                                           selected="boxplot_single_case")
                                        else:
                                            ui.update_navs("boxplot_conditional_ui",
                                                           selected="boxplot_multiple_case")
                                
                                elif dv_type == "Probability plot":
                                    with ui.layout_columns(col_widths=(4, 8)):
                                        inline_label("Column")
                                        ui.input_selectize("proba_plot_selectize", "", choices=[""] + col_nums)
                                        distr_choices = ["Normal", "Exponential", "Uniform"]
                                        inline_label("Distribution")
                                        ui.input_selectize("proba_plot_distri_selectize", "",
                                                           choices=distr_choices)
                                    
                                    ui.input_switch("proba_plot_standardize_switch", "Standardize")

                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Palette", pt="8px")
                                        with ui.layout_columns(col_widths=(6, 6), gap="2px"):

                                            @render.ui
                                            def probplot_hexcolor():
                                                c = color.get()
                                                return ui.HTML(f"<span style='{hc_style}'>{c}</span>")

                                            @render_bokeh(height="36px", width="100%", fill=True)
                                            def probplot_color_picker():
                                                picker = ColorPicker(
                                                    title="",
                                                    color='#1f77b4',
                                                    min_width=95
                                                )
                                            
                                                def update_probplot_picker_color(attr, old, new):
                                                    color.set(new)

                                                picker.on_change('color', update_probplot_picker_color)
                                                return picker

                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("proba_plot_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)
                                    
                                    @reactive.effect
                                    @reactive.event(input.proba_plot_selectize,
                                                    input.proba_plot_standardize_switch)
                                    def proba_plot_labels_update():
                                        if input.proba_plot_selectize() != "":
                                            ui.update_text("fig_xlabel_text", value="Theoretical Quantiles")
                                            std_code = "Standardized " if input.proba_plot_standardize_switch() else ""
                                            ui.update_text("fig_ylabel_text", value=f"{std_code}Sample Quantiles")

                                elif dv_type == "Pair plot":

                                    ui.input_selectize("pair_columns_selectize", "Columns", 
                                                       choices=[""] + col_nums,
                                                       multiple=True, remove_button=True)
                                    ui.input_selectize("pair_drop_rows_selectize", "Drop rows",
                                                       choices=[""], multiple=True, remove_button=True,
                                                       options={"placeholder": "None"})
                                    
                                    @reactive.effect
                                    def pair_columns_selectize_choices_update():
                                        cols = list(input.pair_columns_selectize())
                                        if len(cols) > 0:
                                            ui.update_selectize("pair_drop_rows_selectize", choices=cols)

                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Hues")
                                        hue_choices = [""] + discrete_labels(data, max_cats=8)
                                        ui.input_selectize("pair_hue_selectize", "",
                                                           choices=[""] + hue_choices, remove_button=True,
                                                           options={"placeholder": "None"})
                                        inline_label("Colormap")
                                        ui.input_selectize("pair_cmap_selectize", "",
                                                           choices=cat_cmaps, selected="tab10")
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("pair_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)

                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_selectize("pair_kind_selectize", "Plot kind", 
                                                           choices=["scatter", "kde", "hist", "reg"])
                                        ui.input_selectize("pair_diag_kind_selectize", "Diagonal kind", 
                                                           choices=["auto", "kde", "hist"])
                                    ui.input_switch("pair_corner_switch", "Corner")
                                
                                elif dv_type == "Heat map":
                                    ui.input_selectize("heatmap_columns_selectize", "Columns",
                                                       choices=col_nbs, selected=[], remove_button=True,
                                                       multiple=True)
                                    
                                    with ui.layout_columns(col_widths=(4, 8)):
                                        inline_label("Colormap")
                                        ui.input_selectize("heatmap_colormap_selectize", "",
                                                           choices=num_cmaps)

                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_switch("heatmap_annot_switch", "Annotate", value=True)
                                        ui.input_switch("heatmap_top_tick_switch", "Ticks at top", value=True)

                                elif dv_type == "Bar chart":
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Y-data")
                                        ui.input_selectize("bar_ydata_selectize", "", choices=[""]+col_nums)
                                    
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Palette", pt="8px")
                                        with ui.layout_columns(col_widths=(6, 6), gap="2px"):

                                            @render.ui
                                            def bar_hexcolor():
                                                c = color.get()
                                                return ui.HTML(f"<span style='{hc_style}'>{c}</span>")

                                            @render_bokeh(height="36px", width="100%", fill=True)
                                            def bar_color_picker():
                                                picker = ColorPicker(
                                                    title="",
                                                    color=init_color.get(),
                                                    min_width=95
                                                )
                                            
                                                def update_bar_picker_color(attr, old, new):
                                                    color.set(new)

                                                picker.on_change('color', update_bar_picker_color)
                                                return picker
                                    
                                    with ui.layout_columns(col_widths=(-6, 6)):
                                        ui.input_action_button("bar_add_button", "New bar", )

                                        @reactive.effect
                                        @reactive.event(input.bar_ydata_selectize)
                                        def bar_add_button_disable():

                                            ui.update_action_button("bar_add_button",
                                                                    disabled=input.bar_ydata_selectize() == "")
                                    
                                    hr(1, 0.4)
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_switch("bar_sort_switch", "Sort bars")

                                        @render.express
                                        def bar_sort_ascending_ui():
                                            if input.bar_sort_switch():
                                                ui.input_switch("bar_sort_descending_switch", "Descending")

                                    @render.express
                                    def bar_sort_by_ui():
                                        if input.bar_sort_switch():
                                            choices = []
                                            bars = dv_memory.get().copy()
                                            for bar in bars:
                                                choices.append(bar["ydata"])
                                            if input.bar_ydata_selectize() != "":
                                                choices.append(input.bar_ydata_selectize())
                                            if input.bar_xdata_selectize() != "":
                                                choices.append(input.bar_xdata_selectize())
                                            with ui.layout_columns(col_widths=(3, 9)):
                                                inline_label("Sort by")
                                                ui.input_selectize("bar_sort_by_selectize", "",
                                                                   choices=[""] + choices, remove_button=True,
                                                                   options={"placeholder": "Row index"})
                                    
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("X-data")
                                        ui.input_selectize("bar_xdata_selectize", "",
                                                           choices=[""]+columns, remove_button=True,
                                                           options={"placeholder": "Row index"})
                                    
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        dt = ["Vertical", "Horizontal"]
                                        ui.input_selectize("bar_direction_selectize", "Direction", choices=dt)
                                        btype = ["Clustered", "Stacked"]
                                        ui.input_selectize("bar_mode_selectize", "Type of bars", choices=btype)

                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Width", pt="22px")
                                        ui.input_slider("bar_width_slider", "",
                                                        min=0.1, max=1.0, value=0.8, step=0.05)
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("bar_alpha_slider", "",
                                                        min=0.2, max=1.0, value=1.0, step=0.05)
                                
                                elif dv_type == "Radar chart":
                                    ui.input_selectize("radar_selectize", "Columns",
                                                       choices=[""] + col_nums,
                                                       multiple=True, remove_button=True)
                                    with ui.layout_columns(col_widths=(4, 8)):
                                        inline_label("Ticks angle", pt="22px")
                                        ui.input_slider("radar_tick_angle_slider", "",
                                                        min=0, max=355, value=0, step=5)
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Category")
                                        ui.input_selectize("radar_cats_selectize", "",
                                                           choices=[""] + columns, remove_button=True,
                                                           options={"placeholder": "Row index"})
                                        inline_label("Colormap")
                                        ui.input_selectize("radar_cmap_selectize", "",
                                                           choices=cat_cmaps, selected="tab10")
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("radar_alpha_slider", "",
                                                        min=0.2, max=1.0, value=0.6, step=0.05)

                                elif dv_type == "Line plot":

                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Y-data")
                                        ui.input_selectize("line_ydata_selectize", "",
                                                           choices=[""]+col_nums)
                                        
                                        inline_label("X-data")
                                        ui.input_selectize("line_xdata_selectize", "",
                                                           choices=[""]+columns, remove_button=True,
                                                           options={"placeholder": "Row index"})
                                        
                                        inline_label("Margin")
                                        ui.input_selectize("line_margin_data_selectize", "",
                                                           choices=[""]+col_nums,
                                                           multiple=True, remove_button=True,
                                                           options={"placeholder": "None", "maxItems": 2})
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        styles = ["solid", "dash", "dot", "dash-dot"]
                                        ui.input_selectize("line_style_selectize", "Style", choices=styles)
                                        markers = ["none", "circle", "square", "dot",
                                                   "diamond", "triangle", "star", "cross"]
                                        ui.input_selectize("line_marker_selectize", "Marker", choices=markers)
                                    
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label('Width', pt="22px")
                                        ui.input_slider("line_width_slider", "",
                                                        min=0.5, max=4, step=0.5, value=1.5)

                                        inline_label("Scale", "22px")
                                        ui.input_slider("line_marker_scale_slider", "",
                                                        min=0.1, max=2, step=0.05, value=1)

                                        inline_label("Palette", pt="8px")
                                        with ui.layout_columns(col_widths=(6, 6), gap="2px"):

                                            @render.ui
                                            def line_hexcolor():
                                                c = color.get()
                                                return ui.HTML(f"<span style='{hc_style}'>{c}</span>")

                                            @render_bokeh(height="36px", width="100%", fill=True)
                                            def line_color_picker():
                                                picker = ColorPicker(
                                                    title="",
                                                    color=init_color.get(),
                                                    min_width=95
                                                )
                                            
                                                def update_line_picker_color(attr, old, new):
                                                    color.set(new)

                                                picker.on_change('color', update_line_picker_color)
                                                return picker
                                    
                                    with ui.layout_columns(col_widths=(-6, 6)):
                                        ui.input_action_button("line_add_button", "New line", )
                                    
                                    @reactive.effect
                                    @reactive.event(input.line_ydata_selectize)
                                    def line_add_button_disable():

                                        ui.update_action_button("line_add_button",
                                                                disabled=input.line_ydata_selectize() == "")
                                
                                elif dv_type == "Scatter plot":
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Y-data")
                                        ui.input_selectize("scatter_ydata_selectize", "",
                                                           choices=[""]+columns)
                                        inline_label("X-data")
                                        ui.input_selectize("scatter_xdata_selectize", "",
                                                           choices=[""]+columns, remove_button=True,
                                                           options={"placeholder": "Row index"})
                                        @reactive.effect
                                        @reactive.event(input.scatter_xdata_selectize,
                                                        input.scatter_ydata_selectize)
                                        def scatter_labels_update():
                                            ui.update_text("fig_xlabel_text",
                                                           value=input.scatter_xdata_selectize())
                                            ui.update_text("fig_ylabel_text",
                                                           value=input.scatter_ydata_selectize())

                                        inline_label("Size")
                                        ui.input_selectize("scatter_size_data_selectize", "",
                                                           choices=[""]+col_nums,
                                                           remove_button=True, options={"placeholder": "None"})
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label('Scale', pt="22px")
                                        ui.input_slider("scatter_size_scale_slider", "",
                                                        min=0.1, max=2, value=1, step=0.05)
                                    
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Hues")
                                        ui.input_selectize("scatter_color_data_selectize", "",
                                                           choices=[""]+columns,
                                                           remove_button=True, options={"placeholder": "None"})
                                    
                                    with ui.navset_hidden(id="scatter_conditional_ui"):
                                        with ui.nav_panel(None, value="scatter_single_case"):
                                            with ui.layout_columns(col_widths=(3, 9)):
                                                inline_label("Palette", pt="8px")
                                                with ui.layout_columns(col_widths=(6, 6), gap="2px"):

                                                    @render.ui
                                                    def scatter_hexcolor():
                                                        c = color.get()
                                                        return ui.HTML(f"<span style='{hc_style}'>{c}</span>")

                                                    @render_bokeh(height="36px", width="100%", fill=True)
                                                    def scatter_color_picker():
                                                        picker = ColorPicker(
                                                            title="",
                                                            color='#1f77b4',
                                                            min_width=95
                                                        )
                                            
                                                        def update_scatter_picker_color(attr, old, new):
                                                            color.set(new)

                                                        picker.on_change('color', update_scatter_picker_color)
                                                        return picker
                                        
                                        with ui.nav_panel(None, value="scatter_multiple_case"):
                                            with ui.layout_columns(col_widths=(3, 9)):
                                                inline_label("Colormap")
                                                ui.input_selectize("scatter_cmap_selectize", "",
                                                                   choices=num_cmaps, selected="viridis")
                                        
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("scatter_alpha_slider", "",
                                                        min=0.2, max=1.0, value=1.0, step=0.05)
                                    
                                    @reactive.effect
                                    @reactive.event(input.scatter_color_data_selectize)
                                    def scatter_color_data_selectize_update_ui():

                                        color_data = input.scatter_color_data_selectize()

                                        if color_data == "":
                                            ui.update_navs("scatter_conditional_ui",
                                                           selected="scatter_single_case")
                                        else:
                                            ui.update_navs("scatter_conditional_ui",
                                                           selected="scatter_multiple_case")
                                            
                                            if color_data in col_cats:
                                                cmaps, cmap = cat_cmaps, "tab10"
                                            else:
                                                cmaps, cmap = num_cmaps, "viridis"
                                            ui.update_selectize("scatter_cmap_selectize",
                                                                choices=cmaps, selected=cmap)
                                
                                elif dv_type == "Regression plot":
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Y-data")
                                        ui.input_selectize("regplot_ydata_selectize", "",
                                                           choices=[""]+col_nbs)
                                        inline_label("X-data")
                                        ui.input_selectize("regplot_xdata_selectize", "",
                                                           choices=[""]+col_nbs)
                                        
                                        @reactive.effect
                                        @reactive.event(input.regplot_xdata_selectize,
                                                        input.regplot_ydata_selectize)
                                        def regplot_labels_update():
                                            ui.update_text("fig_xlabel_text",
                                                           value=input.regplot_xdata_selectize())
                                            ui.update_text("fig_ylabel_text",
                                                           value=input.regplot_ydata_selectize())
                                    
                                        choices = [""] + discrete_labels(data, max_cats=8)
                                        inline_label("Hues")
                                        ui.input_selectize("regplot_color_data_selectize", "",
                                                           choices=choices,
                                                           remove_button=True, options={"placeholder": "None"})
                                    
                                    with ui.navset_hidden(id="regplot_conditional_ui"):
                                        with ui.nav_panel(None, value="regplot_single_case"):
                                            with ui.layout_columns(col_widths=(3, 9)):
                                                inline_label("Palette", pt="8px")
                                                with ui.layout_columns(col_widths=(6, 6), gap="2px"):

                                                    @render.ui
                                                    def regplot_hexcolor():
                                                        c = color.get()
                                                        return ui.HTML(f"<span style='{hc_style}'>{c}</span>")

                                                    @render_bokeh(height="36px", width="100%", fill=True)
                                                    def regplot_color_picker():
                                                        picker = ColorPicker(
                                                            title="",
                                                            color='#1f77b4',
                                                            min_width=95
                                                        )
                                            
                                                        def update_regplot_picker_color(attr, old, new):
                                                            color.set(new)

                                                        picker.on_change('color', update_regplot_picker_color)
                                                        return picker
                                        
                                        with ui.nav_panel(None, value="regplot_multiple_case"):
                                            with ui.layout_columns(col_widths=(3, 9)):
                                                inline_label("Colormap")
                                                ui.input_selectize("regplot_cmap_selectize", "",
                                                                   choices=num_cmaps, selected="viridis")
                                    
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("regplot_alpha_slider", "",
                                                        min=0.2, max=1.0, value=1.0, step=0.05)
                                    

                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_switch("regplot_fitted_line_switch", "Fitted line", value=True)
                                        ui.input_switch("regplot_centroid_switch", "Centroid")
                                        inline_label("Confidence level")
                                        ui.input_selectize("regplot_ci_level_selectize", "",
                                                           choices=["None", "80%", "85%", "90%", "95%", "99%"])
                                        inline_label("Transformation")
                                        ui.input_selectize("regplot_transform_selectize", "",
                                                           choices=["None", "Polynomial"])

                                    @render.express(inline=True)
                                    def regplot_poly_degree_ui():
                                        if input.regplot_transform_selectize() == "Polynomial":
                                            with ui.layout_columns(col_widths=(6, 6)):
                                                inline_label("Polynomial order")
                                                ui.input_numeric("regplot_poly_order_numeric", "", 
                                                                 min=2, max=10, step=1, value=2)

                                    @reactive.effect
                                    @reactive.event(input.regplot_ydata_selectize)
                                    def regplot_logistic_choices_update():
                                        node = node_input.get()
                                        data = node["data"]
                                        label = to_selected_columns(input.regplot_ydata_selectize(), data)
                                        if label != "":
                                            yvalue = data[label]
                                            choices = ["None", "Polynomial"]
                                            if (yvalue > 0).all():
                                                choices.append("Log")
                                            if is_bool_dtype(yvalue) or ((yvalue*(1-yvalue) >= 0).all()):
                                                choices.append("Logistic")
                                            ui.update_selectize("regplot_transform_selectize", 
                                                                choices=choices)

                                    @reactive.effect
                                    @reactive.event(input.regplot_color_data_selectize)
                                    def regplot_color_data_selectize_update_ui():

                                        color_data = input.regplot_color_data_selectize()

                                        if color_data == "":
                                            ui.update_navs("regplot_conditional_ui",
                                                           selected="regplot_single_case")
                                        else:
                                            ui.update_navs("regplot_conditional_ui",
                                                           selected="regplot_multiple_case")
                                            cmaps, cmap = cat_cmaps, "tab10"
                                            ui.update_selectize("regplot_cmap_selectize",
                                                                choices=cmaps, selected=cmap)

                                elif dv_type == "Filled areas":

                                    ui.input_selectize("filled_areas_ydata_selectize", "Y-data",
                                                       choices=[""] + col_nums,
                                                       multiple=True, remove_button=True)
                                    
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("X-data")
                                        ui.input_selectize("filled_areas_xdata_selectize", "",
                                                           choices=[""] + columns, remove_button=True,
                                                           options={"placeholder": "Row index"})

                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Style")
                                        ui.input_selectize("filled_areas_style_selectize", "",
                                                           choices=["Layer", "Stack"], selected="Stack")

                                        inline_label("Cmap")
                                        ui.input_selectize("filled_areas_cmap_selectize", "",
                                                           choices=cat_cmaps, selected="tab10")

                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("filled_areas_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)
                                
                                if dv_type == "ACF and PACF":
                                    ui.input_selectize("ac_plot_selectize", "Columns",
                                                       choices=[""] + col_nums,
                                                       multiple=True, remove_button=True,
                                                       options={"maxItems": 8})
                                    max_lags = min([data.shape[0] // 2, 100])
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_selectize("ac_plot_type_selectize", "Plot type",
                                                           choices=["ACF", "PACF"])
                                        ui.input_selectize("ac_plot_method_selectize", "Method", choices=[""])
                                        ui.input_numeric("ac_plot_lags_numeric", "Lags",
                                                         min=4, max=max_lags, value=27, step=1)
                                        ui.input_selectize("ac_plot_ci_selectize", "Confidence level",
                                                           choices=["80%", "85%", "90%", "95%", "99%"],
                                                           selected="95%")

                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Palette", pt="8px")
                                        with ui.layout_columns(col_widths=(6, 6), gap="2px"):
                                    
                                            @render.ui
                                            def ac_plot_hexcolor():
                                                c = color.get()
                                                return ui.HTML(f"<span style='{hc_style}'>{c}</span>")

                                            @render_bokeh(height="36px", width="100%", fill=True)
                                            def ac_plot_color_picker():
                                                picker = ColorPicker(
                                                    title="",
                                                    color='#1f77b4',
                                                    min_width=95
                                                )
                                            
                                                def update_ac_plot_picker_color(attr, old, new):
                                                    color.set(new)

                                                picker.on_change('color', update_ac_plot_picker_color)
                                                return picker

                                    @reactive.effect
                                    @reactive.event(input.ac_plot_type_selectize)
                                    def ac_plot_method_choices_update():

                                        if input.ac_plot_type_selectize() == "ACF":
                                            choices = ["Not adjusted", "Adjusted"]
                                        else:
                                            choices = ["ywunbiased", "yw", "ywm", "ols", "ols-inefficient",
                                                       "ols-adjusted", "ld", "ldb", "burg"]
                                        ui.update_selectize("ac_plot_method_selectize",
                                                            choices=choices, selected=choices[0])

                            if dv_type not in ["Pair plot", "Radar chart", "ACF and PACF"]:
                                with ui.nav_panel("Labels"):
                                    ui.input_text("fig_title_text", "Title")
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_text("fig_xlabel_text", "X-label")
                                        ui.input_text("fig_ylabel_text", "Y-label")
                                            
                                    locs = ["upper left", "upper right",
                                            "lower left", "lower right"]    
                                    ui.input_selectize("fig_legend_loc_selectize", "Legend location",
                                                       choices=locs, selected=locs[0])
        
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_selectize("fig_fontsize_selectize", "Font size",
                                                           choices=list(range(6, 21)), selected=10)
                                        ui.input_numeric("fig_xtick_rotate_numeric", "Rotate X-ticks:",
                                                         min=-90, max=90, step=10, value=0)

                            with ui.nav_panel("Figure"):

                                ui.input_switch("fig_grid_switch", "Grid")
                                            
                                ui.markdown("Figure size")
                                    
                                with ui.layout_columns(col_widths=(3, 9)):
                                    inline_label('Width', '22px')
                                    ui.input_slider("fig_width_slider", "",
                                                    value=640, min=150, max=1500, step=5)
                                    inline_label('Height', '22px')
                                    ui.input_slider("fig_height_slider", "",
                                                    value=480, min=150, max=1500, step=5)
                        
                        hr(0)
                        ui.input_text_area("dv_markdown_text_area", "Markdown",
                                        placeholder="Key in notes...", height="100px")
                        
                    @reactive.effect
                    @reactive.event(input.bar_add_button, ignore_init=True)
                    def bar_add_button_action():
                        
                        bars = dv_memory.get()
                        bars.append(dict(ydata=input.bar_ydata_selectize(), color=color.get()))
                        ui.update_selectize("bar_ydata_selectize", selected="")

                        index = default_colors.index(init_color.get())
                        init_color.set(default_colors[(index + 1) % len(default_colors)])
                        color.set(init_color.get())
                    
                    @reactive.effect
                    @reactive.event(input.line_add_button, ignore_init=True)
                    def line_add_button_action():
                        
                        lines = dv_memory.get()
                        lines.append(dict(ydata=input.line_ydata_selectize(),
                                          xdata=input.line_xdata_selectize(),
                                          margin=input.line_margin_data_selectize(),
                                          color=color.get(),
                                          style=input.line_style_selectize(),
                                          marker=input.line_marker_selectize(),
                                          width=input.line_width_slider(),
                                          scale=input.line_marker_scale_slider()))

                        ui.update_selectize("line_margin_data_selectize", selected="")
                        ui.update_selectize("line_ydata_selectize", selected="")
                        ui.update_selectize("line_style_selectize", selected="solid")
                        ui.update_selectize("line_marker_selectize", selected="none")

                        ui.update_slider("line_width_slider", value=1.5)
                        ui.update_slider("line_marker_scale_slider", value=1)

                        index = default_colors.index(init_color.get())
                        init_color.set(default_colors[(index + 1) % len(default_colors)])
                        color.set(init_color.get())

                ui.input_switch("dv_show_code_switch", "Show code")

                @render.express(inline=True)
                def show_visual_results_source():
                
                    node = node_input.get()
                    data = node["data"]
                    if data is None:
                        return
                    name_in = node["name"]
                    dvs_dict = dvs.get()
                    col_nums, col_cats, col_nbs = num_cat_labels(data)

                    source = visual_source(dv_selected.get(), name_in, data, input, color.get(), dv_memory.get())
                    dvs_dict["type"] = dv_selected.get()
                    dvs_dict["source"] = source

                    if input.dv_show_code_switch():
                        
                        @render.ui
                        @reactive.event(input.dv_markdown_text_area)
                        def dv_markdown_display():
                            if input.dv_markdown_text_area().strip() != "":
                                return ui.markdown(input.dv_markdown_text_area())

                        @render.code
                        def dv_code_display():
                            return (
                                f"{'\n'.join(source['imports'])}\n\n"
                                f"{source['code']}"
                            )

                        hr()

                    with ui.card(height="720px", full_screen=True):
                        fig = visual_exec_source(data, name_in, dvs_dict)
                        if isinstance(fig, str):
                            ui_block(fig, "danger")
                            fig = None
                        else:
                            size = (640, 480) if fig is None else (fig.get_size_inches() * 100).round()
                            width, height = size
                            @render.plot(width=width, height=height)
                            def visual_figure_preview(): 
                                return fig

                            dvs_dict["width"] = width
                            dvs_dict["height"] = height
                            dvs_dict["fig"] = fig

                with ui.layout_columns(col_widths=(2, -8, 2)):
                    ui.input_action_button("cancel_visual_button", "Cancel")
                    ui.input_action_button("save_visual_button", "Save")
                
                @reactive.effect
                @reactive.event(input.save_visual_button)
                def save_visual_button_action():

                    dvs_dict = dvs.get()
                    node_list = nodes.get()
                    cs = node_highlights.get()

                    view = dict(fig=dvs_dict["fig"], width=dvs_dict["width"], height=dvs_dict["height"])
                    root = node_list[cs.index("red")]
                    info = dict(view=view, source=dvs_dict["source"])
                    node_view.set(view)
                    node_list.append(root.grow("visual", info=info))
                    
                    cs = ["gray"] * len(cs) + ["red"]
                    node_highlights.set(cs)
                    arc_highlights.set(["gray"] * len(cs))
                    
                    tool_disable(True)

                @reactive.effect
                @reactive.event(input.cancel_visual_button, input.save_visual_button)
                def visual_to_canvas_action():

                    ui.update_text_area("dv_markdown_text_area", value="")
                    dv_memory.set([])
                    color.set("#1f77b4")
                    init_color.set(default_colors[0])

        with ui.nav_panel(None, value="mds_panel"):
            with ui.layout_sidebar(height="900px"):
                with ui.sidebar(bg='#f8f8f8', width='350px', height='900px'):
                    
                    @render.express
                    def mds_panel_ui():

                        node = node_input.get()
                        data, name = node["data"], node["name"]
                        if data is None:
                            return
                        mds_dict = mds.get()
                        md_type = md_selected.get()

                        col_nums, col_cats, col_nbs = num_cat_labels(data)
                        col_predicted, col_predictors = model_variables(data)
                        if md_type == "Statsmodels":
                            col_predicted = col_nums

                        ui.markdown(f"#### {md_type}")
                        with ui.navset_hidden(id="model_page_navset"):
                            with ui.nav_panel("model_page1"):
                                if md_type == "Sklearn models":
                                    ui.markdown("**Step 1: specify variables**")

                                ui.input_selectize("model_dependent_selectize", "Dependent variable",
                                                   choices=[""] + col_predicted)
                                ui.input_selectize("model_independent_selectize", "Independent variables",
                                                   choices=[""] + col_predictors,
                                                   multiple=True, remove_button=True)
                                ui.input_selectize("model_numeric_cats_selectize", "Numbers treated as categories",
                                                   choices=[""], multiple=True, remove_button=True,
                                                   options={"placeholder": "None"})
                                
                                if md_type == "Statsmodels":
                                    ui.input_selectize("statsmodels_type_selectize", "Model type",
                                                       choices=["ols", "logit"])
                                    ui.input_text("statsmodels_formula_text", "Formula")

                                    hr()
                                    ui.input_text("statsmodels_output_text", "Output name",
                                                  placeholder="Key in a variable name...")

                                    @render.express
                                    def fit_statsmodels_ui():
                                        if input.statsmodels_formula_text().strip() != "":
                                            ui.input_task_button("statsmodels_fitting_button", "Fit model",
                                                                 label_busy="Running...", width="100%")

                            with ui.nav_panel("model_page2"):
                                ui.markdown("**Step 2: model pipeline**")
                                
                                @render.express
                                @reactive.event(input.model_dependent_selectize,
                                                input.model_independent_selectize)
                                def sklearn_predicted_var_ui():
                                    mds_dict = mds.get()
                                    pred_type = mds_dict["type"]
                                    if pred_type == "Classifier":
                                        ind_col = input.model_independent_selectize()
                                        nc = len(set(ind_col).intersection(set(col_cats) - set(col_nbs)))
                                        nc += len(input.model_numeric_cats_selectize())
                                        os_choices = ["Not applied", "RandomOverSampler", "SMOTE"]
                                        if nc == 0:
                                            os_choices.append("ADASYN")
                                        ui.input_selectize("sklearn_over_sampling_selectize", "Over-sampling",
                                                           choices=os_choices)
                                    else:
                                        ui.input_switch("sklearn_predicted_log_switch",
                                                        "Log transformation of target")

                                @render.express
                                def sklearn_over_sampling_k_neighbors_ui():
                                    mds_dict = mds.get()
                                    pred_type = mds_dict["type"]
                                    if pred_type == "Classifier":
                                        if input.sklearn_over_sampling_selectize() in ["SMOTE", "ADASYN"]:
                                            with ui.layout_columns(col_widths=(4, 8)):
                                                inline_label("Neighbors")
                                                ui.input_text("sklearn_over_sampling_k_neighbors", "",
                                                              placeholder="5")

                                with ui.layout_columns(col_widths=(4, 8)):
                                    inline_label("Scalilng")
                                    ui.input_selectize("sklearn_scaling_selectize", "",
                                                       choices=["Not applied", "StandardScaler", "Normalizer"])
                                    inline_label("PCA")
                                    ui.input_text("sklearn_pca_numbers", "", placeholder="Not applied")

                                hr()
                                ui.input_selectize("sklearn_model_selectize", "Model selection",
                                                   choices=[""])
                                
                                @render.express
                                def sklearn_model_hypers_ui():
                                    model_name = input.sklearn_model_selectize()
                                    if model_name == "":
                                        return
                                    hyper_list = model_hypers[model_name]
                                    
                                    if len(hyper_list) == 1:
                                        hyper, label, default_value = hyper_list[0]
                                        ui.input_text(f"sklearn_{model_name.lower()}_{hyper}", label,
                                                      placeholder=default_value)
                                    else:
                                        for hyper, label, default_value in hyper_list:
                                            with ui.layout_columns(col_widths=(6, 6), gap="5px"):
                                                inline_label(label)
                                                ui.input_text(f"sklearn_{model_name.lower()}_{hyper}", "",
                                                              placeholder=default_value)
                    
                            with ui.nav_panel("model_page3"):
                                ui.markdown("**Step 3: model fitting and evaluation**")
                                with ui.layout_columns(col_widths=(4, 8), px="5px"):
                                    inline_label("CV Folds")
                                    ui.input_numeric("sklearn_cv_folds_numeric", "",
                                                     min=2, max=100, step=1, value=5)
                            
                                ui.input_switch("sklearn_test_set_switch", "Test ratio")
                                @render.express
                                def sklearn_test_ratio_shown():
                                    if input.sklearn_test_set_switch():
                                        ui.input_numeric("sklearn_test_ratio_numeric", "",
                                                         min=0.05, max=0.5, step=0.05, value=0.25)
                                
                                ui.input_task_button("sklearn_fitting_button", "Fit model",
                                                     label_busy="Running...", width="100%")

                            with ui.nav_panel("model_page4"):
                                ui.markdown("**Step 4: save results**")
                                ui.input_text("sklearn_output_text", "Output name",
                                              placeholder="Key in a variable name...")
                                
                                @render.express
                                def sklearn_regression_out_ui():
                                    mds_dict = mds.get()
                                    if mds_dict["type"] == "Regressor":
                                        ui.input_switch("sklearn_residual_switch", "Include residuals")
                                
                                ui.input_checkbox_group("sklearn_outputs_checkbox", " ", choices=[])

                                @render.express
                                def sklearn_class_output_ui():
                                    dependent = input.model_dependent_selectize()
                                    if dependent == "":
                                        return
                                    mds_dict = mds.get()
                                    if mds_dict["type"] == "Classifier":
                                        with ui.layout_columns(col_widths=(4, 8), gap="5px"):
                                            if is_bool_dtype(data[dependent]):
                                                class_choices = ["False", "True"]
                                            else:
                                                class_choices = np.unique(data[dependent]).tolist()
                                            inline_label("Target class")
                                            ui.input_selectize("sklearn_class_selectize", "",
                                                               choices=[""] + class_choices, selected="",
                                                               remove_button=True,
                                                               options={"placeholder": "None"})

                                        @render.express
                                        def sklearn_class_threshold_ui():
                                            target_class = input.sklearn_class_selectize()
                                            if target_class != "":
                                                with ui.layout_columns(col_widths=(4, 8), gap="5px"):
                                                    inline_label("Threshold", pt="22px")
                                                    ui.input_slider("sklearn_class_threshold_slider", "",
                                                                    min=0.001, max=0.999, value=0.5, step=0.001)
                                
                                @reactive.effect
                                @reactive.event(input.sklearn_class_selectize)
                                def sklearn_outputs_checkbox_choices_udpate():

                                    target_class = input.sklearn_class_selectize()
                                    if target_class == "":
                                        choices = ["Confusion matrix"]
                                    else:
                                        choices = ["Confusion matrix",
                                                   "Receiver-operating characteristic",
                                                   "Precision-recall", ]
                                    ui.update_checkbox_group("sklearn_outputs_checkbox", choices=choices)
                            
                            @reactive.effect
                            @reactive.event(input.model_dependent_selectize)
                            def model_type_update():
                                data = node_input.get()["data"]
                                mds_dict = mds.get()
                                predicted = input.model_dependent_selectize()
                                if predicted != "":
                                    y = data[predicted]
                                    if (not is_numeric_dtype(y)) or is_bool_dtype(y):
                                        mds_dict["type"] = "Classifier"
                                    else:
                                        mds_dict["type"] = "Regressor"

                        if md_type == "Sklearn models":
                            with ui.layout_columns(col_widths=(5, -2, 5)):
                                ui.input_action_button("sklearn_page_back_button", "Back", disabled=True)
                                ui.input_action_button("sklearn_page_next_button", "Next", disabled=True)

                    #ui.input_action_button("md_debug", "Debug")
                    #@render.code
                    #@reactive.event(input.md_debug)
                    #def md_debug_display():
                    #    return str(mds.get()["source"])

                    @reactive.effect
                    @reactive.event(input.model_dependent_selectize, ignore_init=True)
                    def model_independent_selectize_choices_updated():
                        dep_col = input.model_dependent_selectize()
                        if dep_col != "":
                            node = node_input.get()
                            _, choices = model_variables(node["data"])
                            choices.remove(dep_col)
                            ui.update_selectize("model_independent_selectize",
                                                choices=choices, selected=choices)
                    
                    @reactive.effect
                    @reactive.event(input.model_independent_selectize, ignore_init=True)
                    def model_numeric_cats_selectize_choices_updated():
                        ind_col = input.model_independent_selectize()
                        node = node_input.get()
                        data = node["data"]
                        col_nums, col_cats, col_nbs = num_cat_labels(data)

                        cat_col = []
                        for c in ind_col:
                            if c in col_nums:
                                nc = len(data[to_selected_columns(c, data)].unique())
                                if nc <= 30:
                                    cat_col.append(c)
                        ui.update_selectize("model_numeric_cats_selectize",
                                            choices=[""]+cat_col, selected="")

                    @reactive.effect
                    @reactive.event(input.model_dependent_selectize,
                                    input.model_independent_selectize,
                                    input.model_numeric_cats_selectize, ignore_init=True)
                    def statsmodels_formula_text_update():
                        if md_selected.get() == "Statsmodels":
                            dependent = input.model_dependent_selectize()
                            independents = input.model_independent_selectize()
                            num_cats = input.model_numeric_cats_selectize()
                            independents = [f"C({c})" if c in num_cats else c for c in independents]
                            if dependent != "" and independents != "":
                                formula = f"{dependent} ~ {' + '.join(independents)}"
                                ui.update_text("statsmodels_formula_text", value=formula)

                    @reactive.effect
                    @reactive.event(input.sklearn_page_back_button)
                    def sklearn_page_back_button_action():
                        md_page.set(md_page.get() - 1)

                    @reactive.effect
                    @reactive.event(input.sklearn_page_next_button)
                    def sklearn_page_next_button_action():
                        md_page.set(md_page.get() + 1)

                    @reactive.effect
                    def sklearn_page_update():
                        page = md_page.get()
                        ui.update_navs("model_page_navset",
                                    selected=f"model_page{md_page.get()}")
                        ui.update_action_button("sklearn_page_back_button", disabled=page < 2)

                        if page == 1:
                            predicted = input.model_dependent_selectize()
                            predictors = input.model_independent_selectize()
                            disabled = predicted == "" or len(predictors) == 0
                        elif page == 2:
                            disabled = input.sklearn_model_selectize() == ""
                        elif page == 3:
                            disabled = len(md_memory.get()) == 0
                        else:
                            disabled = True
                        ui.update_action_button("sklearn_page_next_button", disabled=disabled)

                    @reactive.effect
                    def sklearn_model_choices():

                        mds_dict = mds.get()
                        if md_page.get() == 2:
                            if mds_dict["type"] == "Regressor":
                                models = ["LinearRegression", "Ridge", "Lasso",
                                          "DecisionTreeRegressor", "RandomForestRegressor"]
                                output_choices = ["Prediction plot", "Residual plot"]
                            elif mds_dict["type"] == "Classifier":
                                models = ["LogisticRegression",
                                          "DecisionTreeClassifier", "RandomForestClassifier"]
                                output_choices = ["Confusion matrix"]
                            else:
                                models = ["No available model"]
                                output_choices = []

                            selected = input.sklearn_model_selectize()
                            if selected not in models:
                                selected = models[0]
                            ui.update_selectize("sklearn_model_selectize",
                                                choices=models, selected=selected)
                            ui.update_checkbox_group("sklearn_outputs_checkbox",
                                                     choices=output_choices)

                    ui.input_text_area("md_markdown_text_area", "Markdown",
                                       placeholder="Key in notes...", height="100px")
                
                @reactive.effect
                @reactive.event(input.statsmodels_fitting_button)
                def statsmodels_results_update():
                    node = node_input.get()
                    data, name = node["data"], node["name"]
                    md_type = md_selected.get()
                    if md_type != "Statsmodels":
                        return

                    mds_dict = mds.get()
                    source = mds_dict["source"]
                    try:
                        statsmodels_ns = {name: data}
                        exec("\n".join(source["imports"]), statsmodels_ns)
                        exec(source["code"].replace("print", ""), statsmodels_ns)
                        result_summary = eval("result.summary()", statsmodels_ns).__str__()
                        name_save = input.statsmodels_output_text().strip()
                        if name_save != "":
                            invalid = invalid_name(name_save, error=True)
                            if invalid is not False:
                                raise invalid
                        mds_dict["memory"] = dict(result=eval("result", statsmodels_ns))
                    except Exception as err:
                        result_summary = err
                        mds_dict["memory"] = {}
                    mds_dict["results"] = result_summary
                    md_memory.set(dict(summary=result_summary))
                
                @reactive.effect
                @reactive.event(input.sklearn_fitting_button, ignore_init=True)
                def sklearn_fitting_results_update():
                    mds_dict = mds.get()
                    current_imports = mds_dict["source"]["imports"][3]
                    current_code = mds_dict["source"]["code"][3].replace("print", "")
                    test_set = input.sklearn_test_set_switch()
                    try:
                        sklearn_ns = {}
                        exec("\n".join(current_imports), sklearn_ns)
                        memory = mds_dict["memory"]
                        for key, value in memory.items():
                            sklearn_ns[key] = value
                        exec(current_code, sklearn_ns)
                        name_save = input.sklearn_output_text().strip()
                        if name_save != "":
                            invalid = invalid_name(name_save, error=True)
                            if invalid is not False:
                                raise invalid

                        if "search = " in current_code:
                            param_lines = []
                            for p in memory["params"]:
                                best_param_value = eval('search', sklearn_ns).best_params_[p]
                                param_lines.append(f"- {p[p.index('__')+2:]}: {best_param_value}")
                            params_code = (
                                "Best parameters:\n"
                                f"{'\n'.join(param_lines)}\n\n"
                            )
                        else:
                            params_code = ""

                        if test_set:
                            test_result = f"\nTest score: {eval('test_score', sklearn_ns):.4f}"
                        else:
                            test_result = ""

                        result = (
                            f"{params_code}"
                            f"{eval('table', sklearn_ns)}\n\n"
                            f"Cross-validation score: {eval('score', sklearn_ns).mean():.4f}"
                            f"{test_result}"
                        )        
                    except Exception as err:
                        result = err

                    if isinstance(result, str):
                        variables = ["model", "cv"] 
                        if mds_dict["type"] == "Regressor":
                            variables.append("yhat_cv")
                        else:
                            variables.append("proba_cv")
                        
                        if test_set:
                            variables.extend(["x_train", "x_test", "y_train", "y_test"])
                            if mds_dict["type"] == "Regressor":
                                variables.append("yhat_test")
                            else:
                                variables.append("proba_test")
                    
                        for var in variables:
                            mds_dict["memory"][var] = eval(var, sklearn_ns)

                    mds_dict["results"] = result
                    md_memory.set(dict(result=result))
                
                @reactive.effect
                @reactive.event(input.sklearn_test_set_switch)
                def sklearn_test_set_switch_update():
                    if md_page.get() == 3:
                        md_memory.set({})

                ui.input_switch("md_show_code_switch", "Show code")

                @render.express(inline=True)
                def show_model_source_results():
                
                    node = node_input.get()
                    data, name = node["data"], node["name"]
                    mds_dict = mds.get()
                    md_type = md_selected.get()
                    memory = md_memory.get()
                    if data is None:
                        return
                
                    mds_dict["outputs"] = []
                    if md_type == "Statsmodels":
                        source = statsmodels_source(mds_dict, name, input)
                        name_save = input.statsmodels_output_text().strip()
                        if name_save == "":
                            coef_source = dict(type=None, code="", imports=[])
                        else:
                            coef_source = statsmodels_outputs_source(input)
                        mds_dict["outputs"].append(coef_source)
                    else:
                        source = sklearn_model_source(mds_dict, name, data, input, md_page.get())
                        name_save = input.sklearn_output_text().strip()
                        if name_save == "":
                            result_source = dict(type=None, code="", imports=[])
                        else:
                            result_source = sklearn_outputs_source(mds_dict, name, data, input)
                        mds_dict["outputs"].append(result_source)
                        plot_source = sklearn_plots_source(mds_dict, name, data, input, md_page.get())
                        mds_dict["outputs"].extend(plot_source)
                    mds_dict["source"] = source

                    if input.md_show_code_switch():
                        @render.ui
                        @reactive.event(input.md_markdown_text_area)
                        def md_markdown_display():
                            if input.md_markdown_text_area().strip() != "":
                                return ui.markdown(input.md_markdown_text_area())

                        if md_type == "Statsmodels":
                            @render.code
                            def statsmodels_code_display():
                                imports = list(set(source["imports"]))
                                imports.sort(reverse=True)
                                code = source["code"]
                                for out in mds_dict["outputs"]:
                                    if out["code"] != "":
                                        imports += out["imports"]
                                        code += f"\n\n{out['code']}"

                                return (
                                    f"{'\n'.join(imports)}\n\n"
                                    f"{code}"
                                )
                        elif md_type == "Sklearn models":
                            @render.code
                            def sklearn_model_code_display():
                                page = md_page.get()
                                imports = source["imports"][page]
                                code = source["code"][page]
                                if page == 4:
                                    code_segments = [] if code == "" else [code]
                                    for out in mds_dict["outputs"]:
                                        if out["code"] != "":
                                            imports += out["imports"]
                                            #code += f"\n\n{out['code']}"
                                            code_segments.append(out["code"])
                                    
                                    code = "\n\n".join(code_segments)

                                imports = list(set(imports))
                                imports.sort(reverse=True)
                                imports_code = "" if len(imports) == 0 else f"{'\n'.join(imports)}\n\n"
                                return (
                                    f"{imports_code}"
                                    f"{code}"
                                )

                        hr()

                    with ui.card(height="720px", full_screen=True):
                        if md_type == "Statsmodels":
                            @render.ui
                            def model_results_display():                        
                                if "summary" in memory:
                                    result_summary = memory["summary"]
                                    if isinstance(result_summary, str):
                                        return ui.markdown(f"```\n{result_summary}\n```")
                                    else:
                                        return ui_block(f"<b>Error</b>: {result_summary}", 'danger')
                        else:
                            page = md_page.get()
                            columns = data.columns.tolist()
                            if page == 1:
                                row, col = data.shape
                                table_width = len(data.__repr__().split('\n')[0]) * 72 // 96
                                with ui.layout_column_wrap(width=f"{table_width}px",
                                                           fixed_width=True, fill=False, fillable=False):
                                    @render.table()
                                    def model_data_preview():
                                        table = display_table(data, 16).style.format(precision=4)
                                        table.set_caption(f"{row} rows x {col} columns")
                                        
                                        styles = table_styles.copy()
                                        predicted = input.model_dependent_selectize()
                                        if predicted != "":
                                            col_index = columns.index(predicted)
                                            c = "#ffe7e7"
                                            styles.append(dict(selector=f"td.col{col_index}",
                                                               props=[("background-color", c)]))
                                        predictors = input.model_independent_selectize()
                                        c = "#eae7ff"
                                        for p in predictors:
                                            col_index = columns.index(p)
                                            styles.append(dict(selector=f"td.col{col_index}",
                                                               props=[("background-color", c)]))

                                        return table.set_table_styles(styles)
                                    
                                current_imports = mds_dict["source"]["imports"][1]
                                current_code = mds_dict["source"]["code"][1]
                                sklearn_ns = {}
                                if len(current_imports) > 0:
                                    exec('\n'.join(current_imports), sklearn_ns)
                                if current_code != "":
                                    name = node["name"]
                                    #exec(f"{name} = data")
                                    sklearn_ns[name] = data
                                    exec(current_code, sklearn_ns)
                                    mds_dict["memory"]["x"] = eval("x", sklearn_ns)
                                    mds_dict["memory"]["y"] = eval("y", sklearn_ns)
                                    if "to_dummies = " in current_code:
                                        mds_dict["memory"]["to_dummies"] = eval("to_dummies", sklearn_ns)

                            elif page == 2:
                                @render.ui
                                def sklearn_pipeline_display():
                                    current_imports = mds_dict["source"]["imports"][2]
                                    current_code = mds_dict["source"]["code"][2]
                                    if current_code != "":
                                        try:
                                            sklearn_ns = {}
                                            if "to_dummies" in mds_dict["memory"]:
                                                sklearn_ns["to_dummies"] = mds_dict["memory"]["to_dummies"]
                                            exec('\n'.join(current_imports), sklearn_ns)
                                            exec(current_code, sklearn_ns)
                                            mds_dict["memory"]["pipe"] = eval("pipe", sklearn_ns)
                                            if "params = " in current_code:
                                                mds_dict["memory"]["params"] = eval("params", sklearn_ns) 
                                            return ui.HTML(eval("pipe", sklearn_ns)._repr_html_())
                                        except Exception as err:
                                            return ui_block(str(err), "danger")
                            
                            elif page == 3:
                                memory = md_memory.get()
                                if "result" in memory:
                                    result = memory["result"]
                                    if isinstance(result, str):
                                        @render.code
                                        def sklearn_result_message_display():
                                            return result
                                    else:
                                        ui_block(result, "danger")

                            elif page == 4:
                                @render.express
                                def sklearn_plots_display():
                                    memory = md_memory.get()
                                    if mds_dict["type"] == "Classifier":
                                        y_label = input.model_dependent_selectize()
                                        if y_label == "":
                                            return 
                                        target_class = input.sklearn_class_selectize()
                                        if is_bool_dtype(data[y_label]) and target_class in ["True", "False"]:
                                            target_class = eval(target_class)
                                        default = target_class == ""

                                        if not default:
                                            threshold = input.sklearn_class_threshold_slider()
                                            mds_dict["memory"]["threshold"] = threshold
                                            mds_dict["memory"]["target"] = target_class
                                            classes = np.unique(data[y_label]).tolist()
                                            index = classes.index(target_class) if target_class in classes else 0
                                            mds_dict["memory"]["index"] = index
                                            mds_dict["memory"]["y_target"] = (data[y_label] == target_class)
                                        
                                    outputs = mds_dict["outputs"]
                                    if "result" in memory:
                                        if isinstance(memory["result"], str):
                                            @expressify
                                            def sklearn_plot_display(idx):
                                                sklearn_ns = dict(mds=mds, render=render)
                                                exec(
                                                    "@render.plot\n"
                                                    f"def plot_display_fun{idx}():\n"
                                                    "    mds_dict = mds.get()\n"
                                                    f"    if {idx} < len(mds_dict['outputs']):\n"
                                                    f"        return mds_dict['outputs'][{idx}]['fig']",
                                                    sklearn_ns
                                                )
                                                outputs = mds.get()["outputs"]
                                                width, height = outputs[idx]["fig"].get_size_inches() * 100
                                                output_plot(f"plot_display_fun{idx}", width=width, height=height)
                                
                                            with ui.layout_columns(col_widths=(6, 6)):
                                                for idx, out in enumerate(outputs):
                                                    if out["type"] == 'plot':
                                                        sklearn_ns = {}
                                                        define_imports = mds.get()["source"]["imports"][4]
                                                        current_imports = define_imports + out["imports"]
                                                        current_imports.extend(["import pandas as pd",
                                                                                "import numpy as np"])
                                                        define_code = mds.get()["source"]["code"][4]
                                                        current_code = f"{define_code}\n" + out["code"]
                                                        if len(current_imports) > 0:
                                                            exec("\n".join(current_imports), sklearn_ns)
                                                        for key, value in mds_dict["memory"].items():
                                                            #exec(f"{key} = value")
                                                            sklearn_ns[key] = value
                                                        
                                                        exec("\n".join(current_code.split("\n")[:-1]), sklearn_ns)
                                                        out["fig"] = eval("fig", sklearn_ns)
                                                        sklearn_plot_display(idx)
                                    else:
                                        ui.markdown(" ")
                                                        
                with ui.layout_columns(col_widths=(2, -8, 2)):
                    ui.input_action_button("cancel_model_button", "Cancel")
                    ui.input_action_button("save_model_button", "Save", disabled=True)
                
                @reactive.effect
                @reactive.event(input.statsmodels_fitting_button, input.statsmodels_output_text)
                def save_statsmodels_button_disable():
                    
                    mds_dict = mds.get()
                    memory = md_memory.get()
                    fit = len(mds_dict["memory"]) > 0
                    invalid = invalid_name(input.statsmodels_output_text().strip())
                    disabled = invalid or (not fit)
                    ui.update_action_button("save_model_button", disabled=disabled)
                
                @reactive.effect
                @reactive.event(input.sklearn_fitting_button, input.sklearn_output_text,
                                input.sklearn_test_set_switch)
                def save_sklearn_button_disable():
                    
                    memory = md_memory.get()
                    disabled = True
                    if "result" in memory:
                        invalid = invalid_name(input.sklearn_output_text().strip())
                        disabled = (not isinstance(memory["result"], str)) or (md_page.get() < 3) or invalid
                    
                    ui.update_action_button("save_model_button", disabled=disabled)

                @reactive.effect
                @reactive.event(input.cancel_model_button, input.save_model_button)
                def model_to_canvas_action():
                    ui.update_text_area("md_markdown_text_area", value="")
                    md_page.set(1)
                    md_memory.set({})
                
                @reactive.effect
                @reactive.event(input.save_model_button)
                def save_model_button_action():

                    node = node_input.get()
                    data, name = node["data"], node["name"]

                    node_list = nodes.get()
                    cs = node_highlights.get()
                    root = node_list[cs.index("red")]

                    # Save model node 
                    mds_dict = mds.get()
                    md_type = md_selected.get()
                    if md_type == "Statsmodels":
                        method = input.statsmodels_type_selectize()
                        model_view = dict(name=f"{md_type}: {method}", 
                                        results=mds_dict["results"])
                        source = mds_dict["source"]
                        define_imports = []
                        define_code = ""
                    else:
                        source = mds_dict["source"]
                        model = input.sklearn_model_selectize()
                        model_view = dict(name=f"{md_type}: {model}", 
                                        results=mds_dict["results"])

                        code = "\n\n".join([seg for seg in source["code"].values() if seg != ""])
                        imports_dict = source["imports"]
                        imports = imports_dict[1] + imports_dict[2] + imports_dict[3] + imports_dict[4]
                        markdown = source["markdown"]
                        source = dict(code=code, imports=imports, markdown=markdown)

                        define_imports = mds_dict["source"]["imports"][4]
                        define_code = mds_dict["source"]["code"][4]
                        
                    node_view.set(model_view)
                    model_info = dict(type=md_type, data=data, view=model_view, source=source)
                    model_node = root.grow("model", info=model_info)
                    
                    node_list.append(model_node)
                    
                    sklearn_ns = {}
                    for key, value in mds_dict["memory"].items():
                        #exec(f"{key} = value")
                        sklearn_ns[key] = value
                    sklearn_ns[name] = data

                    if define_imports:
                        exec("\n".join(define_imports), sklearn_ns)
                    if define_code != "":
                        exec(define_code, sklearn_ns)

                    output_nodes = []
                    for out in mds_dict["outputs"]:
                        if out["type"] == "data":
                            if len(out["imports"]) > 0:
                                exec("\n".join(out['imports']), sklearn_ns)
                            exec(out["code"], sklearn_ns)

                            name_out = out["name_out"]
                            data_out = eval(name_out, sklearn_ns)
                            view = dict(name=name_out, string=data_out.to_string(max_rows=6, max_cols=6),
                                        shape=data_out.shape)
                            info = dict(name=name_out, data=data_out, view=view,
                                        source=dict(code=out["code"], imports=out["imports"], markdown=""))
                            output_nodes.append(model_node.grow("data", info=info))
                            
                            all_names = var_names.get()
                            all_names.append(name_out)
                        elif out["type"] == "plot":
                            fig = out["fig"]
                            view = dict(fig=fig, width=420, height=420)
                            info = dict(view=view, 
                                        source=dict(code=out["code"], imports=out["imports"], markdown=""))
                            output_nodes.append(model_node.grow("visual", info=info))

                    node_list.extend(output_nodes)
                    cs = ['gray'] * len(node_list)
                    cs[-1 - len(output_nodes)] = "red"
                    node_highlights.set(cs)
                    arc_highlights.set(["gray"] * len(cs))

                    tool_disable(True)

        save_buttons = input.save_data_button, input.save_visual_button, input.save_model_button
        cancel_buttons = input.cancel_data_button, input.cancel_visual_button, input.cancel_model_button
        @reactive.effect
        @reactive.event(*(save_buttons + cancel_buttons))
        def node_to_canvas_panel():

            ui.update_navs("main", selected="canvas_panel")
            ops.set(dict(type=None, source=None, data_out=None))
            dvs.set(dict(type=None, source=None, fig=None, width=640, height=480))
            mds.set(dict(type="", source={}, results=None, outputs=None, memory={}))
            
            xmax, ymin = canvas_lim.get()
            node_list = nodes.get()
            node_xmax = max([n.pos[0] for n in node_list])
            node_ymin = min([n.pos[1] for n in node_list])

            if xmax <= node_xmax:
                xmax = node_xmax + 4
            if ymin >= node_ymin:
                ymin = node_ymin - 3.2
            canvas_lim.set((xmax, ymin))
