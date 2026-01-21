#app.py - Shiny app file for running the application.

#The file specifies the design of all UI components used in the application.

#Author:  Peng Xiong (xiongpengnus@gmail.com)
#Date:    March 29, 2025

#Version: 1.0
#License: MIT License

import io
import json
import uuid
from PIL import Image
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from shiny import reactive, req
from shiny.ui import output_plot
from shiny.ui import tags as ui_tags
from shiny.express import render, ui, input, expressify
from htmltools import Tag, tags, HTML

from pandas.api.types import is_string_dtype, is_datetime64_dtype

from .toolset import *
from .autosource import *
from .canvas import *
from .styles import *
from .components import color_input as ui_color_input


ui.head_content(
    ui.tags.script(src="color_binding.js?v=1")  # bump v= to bust caches if you edit the file
)

def hr(margin=0.75, offset=0):

    return ui.HTML(f"<hr style='margin-bottom:{margin + offset}em;margin-top:{margin - offset}em'>")

def shift(pt="-15px"):

    return ui.HTML(f'<div style="margin-top:{pt}"> </div>')

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

    id = name.lower().replace(' ', '_').replace('-', '_')

    return (
        "@reactive.effect\n"
        f"@reactive.event(input.{id}_button)\n"
        f"def to_{id}_section():\n"
        f"    {cat}_selected.set('{name}')\n"
        f"    ui.update_navset('main', selected='{cat}s_panel')\n"
        f"    {cat}_memory.set([])\n"
    )


def tool_disable(disabled):

    for item in ops_menu + dvs_menu + mds_menu:
        item_id = item.lower().replace(' ', '_').replace('-', '_')
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

def default_name(used):

    index = 0
    while True:
        index += 1
        name = f"df{index}"
        if name not in used:
            return name


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
ops_menu = list(ops_menu_dict.keys())

# The menu of all data visualization tools
dvs_menu = list(dvs_menu_dict.keys())

# The menu of all modeling tools
#mds_menu = ["Statsmodels", "Scikit-learn models"]
mds_menu = list(mds_menu_dict.keys())

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

ui.tags.style("""
.popover {
    min-width: 450px !important;
    max-width: 650px !important;
    padding-right: 10px !important;
    padding-left: 10px !important;
}
""")

ui.include_js(Path(__file__).parent / "www/copy_handler.js")

with ui.layout_column_wrap(width="1060px", fixed_width=True):
    with ui.navset_hidden(id="main"):
        with ui.nav_panel(None, value="canvas_panel"):
            ui.HTML('<br>')
            with ui.layout_sidebar(height='900px'):
                with ui.sidebar(width='350px', open="always", bg='#f8f8f8', height="900px"):
                    with ui.navset_tab(id="main_toolset_navs"):
                        button_gap = "0px"
                        button_heights = "83px"
                        icon_size = "95px", "78px"
                        figsize = (4, 3)

                        with ui.nav_panel("Operations", value="ops_toolset_nav"):
                            tool_ns = globals()
                            for op_cat, op_names in op_cats.items():
                                shift("-10px")
                                with ui.card(style=icon_card_style):
                                    ui.card_header((shift("-8px"), op_cat), style=icon_card_header_style)
                                shift("5px")    
                                with ui.layout_columns(col_widths=(4, 4, 4),
                                                       gap=button_gap, row_heights=button_heights):
                                    for op_name in op_names:
                                        op_id = op_name.lower().replace(' ', '_').replace('-', '_')
                                        exec(tool_icon_code(op_id, figsize), tool_ns)
                                        icon = output_plot(f"{op_id}_button_icon",
                                                           width=icon_size[0], height=icon_size[1])
                                        ui.input_action_button(f"{op_id}_button", icon, width="100px",
                                                               style=icon_button_style, disabled=True)
                                        exec(tool_effect_code(op_name, "op"), tool_ns)
                                    
                                    empty_slots = (-len(op_names)) % 3
                                    for _ in range(empty_slots):
                                        ui.HTML("")
                                
                        with ui.nav_panel("Visuals", value="dvs_toolset"):
                            tool_ns = globals()
                            for dv_cat, dv_names in dv_cats.items():
                                shift("-10px")
                                with ui.card(style=icon_card_style):
                                    ui.card_header((shift("-8px"), dv_cat), style=icon_card_header_style)
                                shift("5px") 
                                with ui.layout_columns(col_widths=(4, 4, 4),
                                                       gap=button_gap, row_heights=button_heights):
                                    for dv_name in dv_names:
                                        dv_id = dv_name.lower().replace(' ', '_').replace('-', '_')
                                        exec(tool_icon_code(dv_id, figsize), tool_ns)
                                        icon = output_plot(f"{dv_id}_button_icon",
                                                           width=icon_size[0], height=icon_size[1])
                                        ui.input_action_button(f"{dv_id}_button", icon, width="100px",
                                                               style=icon_button_style, disabled=True)
                                        exec(tool_effect_code(dv_name, "dv"), tool_ns)

                                    empty_slots = (-len(dv_names)) % 3
                                    for _ in range(empty_slots):
                                        ui.HTML("")

                        with ui.nav_panel("Models", value="mds_toolset"):
                            tool_ns = globals()
                            for md_cat, md_names in md_cats.items():
                                shift("-10px")
                                with ui.card(style=icon_card_style):
                                    ui.card_header((shift("-8px"), md_cat), style=icon_card_header_style)
                                shift("5px") 
                                with ui.layout_columns(col_widths=(4, 4, 4),
                                                       gap=button_gap, row_heights=button_heights):
                                    for md_name in md_names:
                                        md_id = md_name.lower().replace(' ', '_').replace('-', '_')
                                        exec(tool_icon_code(md_id, figsize), tool_ns)
                                        icon = output_plot(f"{md_id}_button_icon",
                                                           width=icon_size[0], height=icon_size[1])

                                        ui.input_action_button(f"{md_id}_button", icon, width="100px",
                                                               style=icon_button_style, disabled=True)
                                        exec(tool_effect_code(md_name, "md"), tool_ns)

                                    empty_slots = (-len(md_names)) % 3
                                    for _ in range(empty_slots):
                                        ui.HTML("")

                with ui.layout_columns(col_widths=(5, 7), gap="20px", height="160px"):
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
                    
                    with ui.card(style="min-width:400px"):
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
                
                with ui.card(height='775px', style="min-width:680px"):
                        @render.express
                        def canvas_plot_func():

                            xmax, ymin = canvas_lim.get()
                            canvas_width, canvas_height = int((xmax + 4) * 12.5), int((2.1 - ymin) * 12.5)
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
                            with ui.card(full_screen=True, max_height="550px"):
                                ui.card_header("Dataset", style=chd_style)
                                row, col = view["shape"]
                                @render.code
                                def data_view_code():
                                    return f"{view['name']}: {row} rows x {col} columns\n\n{view['string']}"
                                ui.card_footer(ui.input_action_button("close_data_view", "Close", width="110px"))

                    elif "width" in view and "height" in view:
                        fig = view["fig"]
                        fig.set_dpi(60)
                        width = int(view["width"]*3/5)
                        height = int(view["height"]*3/5)
                        with ui.panel_absolute(draggable=True, width=f"{width + 35}px", **pos):
                            with ui.card(full_screen=True, max_height="550px"):
                                ui.card_header("Figure", style=chd_style)
                                @render.plot(width=width, height=height)
                                def fig_view_plot():
                                    return fig
                                ui.card_footer(ui.input_action_button("close_fig_view", "Close", width="110px"))
                    elif "code" in view:
                        with ui.panel_absolute(draggable=True, width=f"550px", **pos):
                            with ui.card(full_screen=True, max_height="550px"):
                                ui.card_header("Source", style=chd_style)
                                code = view["code"]
                                if isinstance(code, dict):
                                    keys = ["vars", "dummy", "pipeline", "fitting"]
                                    code = '\n'.join([code[k] for k in keys])
                                #clines = code.split("\n")
                                #if len(clines) > 15:
                                #    clines = clines[:6] + ["... ..."] * 3 + clines[-6:]
                                #    code_display = "\n".join(clines)
                                #else:
                                #    code_display = code
                                code_display = code
                                ui.markdown(view["markdown"])
                                hr(-0.5, -0.5)

                                with ui.tags.div(style=copy_button_div_style):
                                    ui.tags.a(
                                        ui.tags.img(src="clipboard.svg",
                                                    style="width:18px;margin:0px;padding:0px"),
                                        role="button",
                                        title="copy",
                                        onclick=(f"copyText(this, {json.dumps(code)}); return false;"),
                                        style="padding:0px;text-decoration:none;"
                                    )
                                @render.code
                                def source_view_code():
                                    return code_display

                                ui.card_footer(ui.input_action_button("close_source_view", "Close", width="110px"))
                    elif "results" in view:
                        with ui.panel_absolute(draggable=True, width=f"650px", **pos):
                            with ui.card(full_screen=True, max_height="550px"):
                                ui.card_header("Model", style=chd_style)
                                results = view["results"]
                                
                                if "estimator" in view:
                                    estimator = view["estimator"]
                                    ui.HTML(estimator._repr_html_())
                                
                                @render.code
                                def model_view_code():
                                    return results

                                ui.card_footer(ui.input_action_button("close_model_view", "Close", width="110px"))

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
                        aggs = ["count", "mean", "median", "std", "var", "min", "max", "sum"]
                        aggs_default = ["mean"]

                        op_type = op_selected.get()

                        with ui.card(style="background-color:#f8f8f8;border:none;box-shadow:none"):
                            with ui.card_header(style="padding:5px;border:none"):
                                ui.span(f"{op_type}", style="font-size:16pt")
                                ui.HTML("&nbsp;&nbsp;&nbsp;")
                                with ui.popover(id="ops_popover", placement="right"):
                                    question_circle_fill
                                    ui.HTML(doc_html(ops_menu_dict[op_type]))

                        if op_type == "Value counts operations":
                            count_choices = columns
                            ui.input_selectize("counts_ops_selectize", "Columns",
                                               choices=count_choices, selected=[],
                                               multiple=True)
                            
                            ui.input_selectize("counts_ops_unstack_selectize", "Unstack levels",
                                               choices=[], selected=[],
                                               multiple=True, remove_button=True),

                            @reactive.effect
                            def counts_ops_unstack_update_choices():
                                selected = list(input.counts_ops_selectize())
                                maxItems = len(selected) - 1 if len(selected) > 1 else 0
                                ui.update_selectize("counts_ops_unstack_selectize",
                                                    choices=selected, selected=[],
                                                    options={"placeholder": "None",
                                                             "maxItems": maxItems,
                                                             "plugins": ["remove_button"]})

                            with ui.layout_columns(col_widths=(6, 6), gap="10px"):
                                @render.express(inline=True)
                                def counts_ops_sort_by_ui_left():
                                    unstack = list(input.counts_ops_unstack_selectize())
                                    if len(unstack) == 0:
                                        ui.input_switch("counts_ops_sort_switch",
                                                        "Sort", value=True)
                                
                                @render.express(inline=True)
                                def counts_ops_sort_by_ui_right():
                                    unstack = list(input.counts_ops_unstack_selectize())
                                    if len(unstack) == 0:
                                        ui.input_switch("counts_ops_sort_descending_switch",
                                                        "Descending", value=True)
                                    else:
                                        ui.HTML("")

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
                            with ui.layout_columns(col_widths=(7, 5, 12, 5, 7), gap="10px"):
                                ui.input_selectize("filter_column_selectize", "Target variable",
                                                   choices=[""] + columns)
                                filter_operators = ["", "==", "!=", "<=", "<", ">=", ">", "in", "not in"]
                                ui.input_selectize("filter_operator_selectize", "Operator",
                                                   choices=filter_operators, selected="")
                            
                                @render.express
                                @reactive.event(input.filter_column_selectize)
                                def filter_value_text_ui():
                                    show_filter_value = True
                                    col = input.filter_column_selectize()
                                    if col in col_nbs and col not in col_nums:
                                        ui.HTML("")
                                    else:
                                    #    show_filter_value = False
                                    #if show_filter_value:
                                        ui.input_text("filter_value_text", "Value(s) to compare")

                                ui.HTML("")
                                ui.input_action_button("add_filter_button", "New bool")

                            @reactive.effect
                            @reactive.event(input.filter_column_selectize)
                            def filter_operator_selectize_update():
                                col = input.filter_column_selectize()
                                if col != "":
                                    if col in col_nbs and col not in col_nums:
                                        filter_operators = ["", "is True", "is False"]
                                    else:
                                        filter_operators = ["", "==", "!=", "<=", "<", ">=", ">", "in", "not in"]
                                    ui.update_selectize("filter_operator_selectize", choices=filter_operators)

                            #with ui.layout_columns(col_widths=(5, 7)):
                            #    ui.HTML("")
                            #    ui.input_action_button("add_filter_button", "New bool")
                            
                            @reactive.effect
                            @reactive.event(input.filter_column_selectize,
                                            input.filter_operator_selectize,
                                            input.filter_value_text)
                            def add_filter_button_disable():
                                cond1 = input.filter_column_selectize() == ""
                                cond2 = input.filter_operator_selectize() == ""
                                cond3 = False
                                if not cond1:
                                    col = input.filter_column_selectize()
                                    if not (col in col_nbs and col not in col_nums):
                                        cond3 = str_to_values(input.filter_value_text(), sup=True) is None
                                ui.update_action_button("add_filter_button", disabled=(cond1 or cond2 or cond3))
                            
                            with ui.layout_columns(col_widths=(6, 6)):
                                ui.input_switch("filter_select_rows_switch", "Filter rows", value=True)

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
                            ui.input_selectize("group_view_columns_selectize", "Data values",
                                               choices=[""] + columns,
                                               multiple=True, remove_button=True)
                            ui.input_selectize("group_methods_selectize", "Methods",
                                               choices=[""] + aggs,
                                               multiple=True, remove_button=True)
                            with ui.layout_columns(col_widths=(6, 6)):
                                ui.input_switch("group_reset_switch", "Reset index")
                                ui.input_switch("group_transpose_switch", "Transpose")
                        elif op_type == "Pivot table":
                            ui.input_selectize("pivot_values_selectize", "Data values",
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
                            ui.input_selectize("nan_method_selectize", "Method", choices=["drop", "fill", "mark"])
                            @render.express
                            def nan_conditional_ui():
                                if input.nan_method_selectize() == "fill":
                                    ui.input_text("nan_fill_value_text", "Value to fill")
                                elif input.nan_method_selectize() == "mark":
                                    ui.input_text("nan_mark_value_label", "To column")
                                elif input.nan_method_selectize() == "drop":
                                    ui.input_switch("nan_reset_switch", "Reset index")
                        elif op_type == "Time trend":
                            with ui.layout_columns(col_widths=(5, 7, 12), gap="10px"):
                                ui.input_checkbox("time_trend_select_all_checkbox", "Select all")
                                ui.HTML("")
                                ui.input_selectize("time_trend_columns_selectize", "Columns",
                                                   choices=[""] + col_nums,
                                                   multiple=True, remove_button=True)
                    
                            with ui.layout_columns(col_widths=(4, 8), gap="10px"):
                                transforms =  ["change", "relative change", "log change",
                                               "moving average", "moving median",
                                               "moving min", "moving max", "moving variance"]
                                inline_label("Transform")
                                ui.input_selectize("time_trend_transform_selectize", "",
                                                   choices=[""] + transforms)
                                inline_label("Steps")
                                ui.input_text("time_trend_steps_text", "", placeholder="1")
                            ui.input_switch("time_trend_drop_original_data", "Drop original data")

                            @reactive.effect
                            @reactive.event(input.time_trend_select_all_checkbox, ignore_init=True)
                            def time_trend_columns_update():
                                if input.time_trend_select_all_checkbox():
                                    ui.update_selectize("time_trend_columns_selectize", selected=col_nbs)
                            
                            @reactive.effect
                            @reactive.event(input.time_trend_columns_selectize)
                            def clustering_select_all_update():
                                node = node_input.get()
                                data_in = node["data"]
                                col_nums, col_cats, col_nbs = num_cat_labels(data_in)
                                columns = input.time_trend_columns_selectize() 
                                if len(columns) < len(col_nums):
                                    ui.update_checkbox("time_trend_select_all_checkbox", value=False)
                        
                        elif op_type == "Date time":
                            tcols = [c for c in columns if
                                     is_datetime64_dtype(data_in[to_selected_columns(c, data_in)]) or
                                     is_timedelta64_dtype(data_in[to_selected_columns(c, data_in)]) or
                                     is_string_dtype(data_in[to_selected_columns(c, data_in)])]
                            with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                inline_label("Column")
                                ui.input_selectize("date_time_column_selectize", "",
                                                   choices=[""] + tcols)
                                
                                with ui.navset_hidden(id="date_time_format_label_navset"):
                                    with ui.nav_panel(None, value="date_time_format_empty_label"):
                                        None
                                    with ui.nav_panel(None, value="date_time_format_label"):
                                        inline_label("Format")
                                
                                with ui.navset_hidden(id="date_time_format_ui_navset"):
                                    with ui.nav_panel(None, value="data_time_format_empty_ui"):
                                        None
                                    with ui.nav_panel(None, value="date_time_format_ui"):
                                        ui.input_text("date_time_format_text", "", placeholder="None")
                            
                            with ui.navset_hidden(id="date_time_td_switch_navset"):
                                with ui.nav_panel(None, value="date_time_td_empty"):
                                    shift()
                                with ui.nav_panel(None, value="date_time_td_ui"):
                                    shift()
                                    with ui.layout_columns(col_widths=(7, 5)):
                                        ui.input_switch("date_time_td_switch", "To duration")
                                        @render.express
                                        @reactive.event(input.date_time_td_switch)
                                        def date_time_duration_reverse_ui():
                                            if input.date_time_td_switch():
                                                ui.input_switch("date_time_duration_reverse_switch", "Reverse")

                            shift("-50px")
                            @render.express(inline=True)
                            @reactive.event(input.date_time_td_switch,
                                            input.date_time_duration_reverse_switch)
                            def date_time_calendar_clock_ui():
                                if input.date_time_td_switch():
                                    is_start = input.date_time_duration_reverse_switch()
                                    time_label = "Start time of duration" if is_start else "End time of duration"
                                    with ui.layout_columns(col_widths=(12, 3, 9, 4, 4, 4), gap="10px"):
                                        ui.markdown(time_label)
                                        inline_label("Calendar")
                                        ui.input_date("date_time_calendar", "")
                                        ui.input_numeric("date_time_clock_hour_numerics", "Hour",
                                                         min=0, max=23, value=0, step=1)
                                        ui.input_numeric("date_time_clock_minute_numerics", "Minute",
                                                         min=0, max=59, value=0, step=1)
                                        ui.input_numeric("date_time_clock_second_numerics", "Second",
                                                         min=0, max=59, value=0, step=1)

                            shift(pt="-50px")
                            with ui.layout_columns(col_widths=(12, 4, 8), gap="10px"):
                                ui.input_selectize("date_time_to_columns_selectize", "Results",
                                                   choices=["timestamp"], selected=["timestamp"],
                                                   multiple=True, remove_button=True)
                                inline_label("Label prefix")
                                ui.input_text("date_time_to_columns_prefix_text", "")
                            
                            @reactive.effect
                            def date_time_to_column_selectize_choices_update():
                                col = to_selected_columns(input.date_time_column_selectize(), data_in)
                                if col in data_in.columns:
                                    if is_timedelta64_dtype(data_in[col]):
                                        date_time_choices = ["duration", "days", "hours", "minutes", "seconds"]
                                        ui.update_navset("date_time_format_label_navset",
                                                         selected="date_time_format_empty_label")
                                        ui.update_navset("date_time_format_ui_navset",
                                                         selected="data_time_format_empty_ui")
                                        ui.update_navset("date_time_td_switch_navset",
                                                         selected="date_time_td_empty")
                                    else:
                                        if input.date_time_td_switch():
                                            date_time_choices = ["duration", "days", "hours", "minutes", "seconds"]
                                        else:
                                            date_time_choices = ["timestamp",
                                                                 "year", "month", "month_name", "day",
                                                                 "hour", "minute", "second"]

                                        ui.update_navset("date_time_format_label_navset",
                                                         selected="date_time_format_label")
                                        ui.update_navset("date_time_format_ui_navset",
                                                         selected="date_time_format_ui")
                                        ui.update_navset("date_time_td_switch_navset",
                                                         selected="date_time_td_ui")
                            
                                    ui.update_selectize("date_time_to_columns_selectize",
                                                        choices=date_time_choices,
                                                        selected=date_time_choices[0])

                        elif op_type == "ANOVA":
                            with ui.layout_columns(col_widths=(12, 12, 12), gap="10px"):
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
                            with ui.layout_columns(col_widths=(5, 2, 5, 12, 12), gap="10px"):
                                ui.input_checkbox("clustering_select_all_checkbox", "Select all")
                                ui.HTML("")
                                ui.input_switch("clustering_dropna_switch", "Drop NA")
                            
                                cluster_columns = list(set(col_nbs + discrete_labels(data_in, max_cats=50)))
                                ui.input_selectize("clustering_columns_selectize", "Features for clustering",
                                                   choices=cluster_columns, selected=[],
                                                   multiple=True, remove_button=True)
                                ui.input_selectize("clustering_numeric_cats_selectize",
                                                   "Numbers treated as categories",
                                                   choices=[""], multiple=True, remove_button=True)
                            
                            with ui.layout_columns(col_widths=(4, 8), gap="10px"):
                                inline_label("Method")
                                ui.input_selectize("clustering_method_selectize", "",
                                                   choices=["K-means clustering", "Hierarchical clustering"])
                                inline_label("Cluster No.")
                                ui.input_text("clustering_numbers_text", "")
                                inline_label("Label prefix")
                                ui.input_text("clustering_label_prefix_text", "", placeholder="cluster_num")
                                inline_label("Value prefix")
                                ui.input_text("clustering_value_prefix_text", "", placeholder="c")

                            @reactive.effect
                            @reactive.event(input.clustering_select_all_checkbox, ignore_init=True)
                            def clustering_columns_update():
                                if input.clustering_select_all_checkbox():
                                    data_in = node_input.get()["data"]
                                    cluster_columns = list(set(col_nbs + discrete_labels(data_in, max_cats=50)))
                                    ui.update_selectize("clustering_columns_selectize",
                                                        selected=cluster_columns)
                            
                            @reactive.effect
                            @reactive.event(input.clustering_columns_selectize)
                            def clustering_select_all_update():
                                node = node_input.get()
                                data_in = node["data"]
                                columns = to_selected_columns(input.clustering_columns_selectize(), data_in)    
                                cluster_columns = set(col_nbs + discrete_labels(data_in, max_cats=50))
                                if len(columns) < len(cluster_columns):
                                    ui.update_checkbox("clustering_select_all_checkbox", value=False)
                                
                                cat_col = []
                                for c in columns:
                                    if c in col_nums:
                                        nc = len(data_in[to_selected_columns(c, data_in)].unique())
                                        if nc <= 30:
                                            cat_col.append(c)
                                ui.update_selectize("clustering_numeric_cats_selectize", choices=cat_col)

                        elif op_type == "Decomposition":
                            with ui.layout_columns(col_widths=(5, 2, 5, 12, 12), gap="10px"):
                                ui.input_checkbox("decomposition_select_all_checkbox", "Select all")
                                ui.HTML("")
                                ui.input_switch("decomposition_dropna_switch", "Drop NA")

                                deco_columns = list(set(col_nbs + discrete_labels(data_in, max_cats=50)))
                                ui.input_selectize("decomposition_columns_selectize", "Features for decomposition",
                                                   choices=deco_columns, selected=[],
                                                   multiple=True, remove_button=True)
                                ui.input_selectize("decomposition_numeric_cats_selectize",
                                                   "Numbers treated as categories",
                                                   choices=[""], multiple=True, remove_button=True)
        
                            with ui.layout_columns(col_widths=(4, 8), gap="10px"):
                                inline_label("Scaling")
                                ui.input_selectize("decomposition_scaling_selectize", "",
                                                   choices=["Not applied", "StandardScaler", "Normalizer"],
                                                   selected="StandardScaler")
                                deco_methods = ["PCA", "KernelPCA", "NMF", "FactorAnalysis"]
                                inline_label("Method")
                                ui.input_selectize("decomposition_method_selectize", "",
                                                   choices=[""] + deco_methods)
                                
                                with ui.navset_hidden(id="decomposition_params_label"):
                                    with ui.nav_panel(None, value="empty_label"):
                                        None
                                    with ui.nav_panel(None, value="kernelpca_label"):
                                        inline_label("Kernel", pt="10px")
                                        @render.express
                                        @reactive.event(input.decomposition_kernels_selectize)
                                        def deco_kernel_degree_label():
                                            if input.decomposition_kernels_selectize() == "poly":
                                                inline_label("Degree", pt="22px")

                                with ui.navset_hidden(id="decomposition_params_ui"):
                                    with ui.nav_panel(None, value="empty_ui"):
                                        None
                                    with ui.nav_panel(None, value="kernelpca_ui"):
                                        kernels = ["linear", "poly", "rbf", "sigmoid", "cosine"]
                                        ui.input_selectize("decomposition_kernels_selectize", "",
                                                           choices=kernels)
                                        @render.express
                                        @reactive.event(input.decomposition_kernels_selectize)
                                        def deco_kernel_degree_ui():
                                            if input.decomposition_kernels_selectize() == "poly":
                                                ui.input_slider("decomposition_poly_kernel_degree", "",
                                                                min=1, max=10, value=3, step=1)
    
                                inline_label("Label prefix")
                                ui.input_text("decomposition_label_prefix_text", "", placeholder="pc")
                                inline_label("Show first", pt="22px")
                                ui.input_slider("decomposition_max_nc_slider", "",
                                                min=1, max=5, value=5, step=1)
                                
                            ui.input_switch("decomposition_replace_feature_switch",
                                            "Replace original features")

                            @reactive.effect
                            @reactive.event(input.decomposition_columns_selectize)
                            def decomposition_max_nc_slider_update():
                                node = node_input.get()
                                data_in = node["data"]
                                columns = to_selected_columns(input.decomposition_columns_selectize(), data_in)    
                                if len(columns) > 0:
                                    num_columns = len(pd.get_dummies(data_in[columns], drop_first=True).columns)
                                    max_comps = max([5, num_columns])
                                    ui.update_slider("decomposition_max_nc_slider",
                                                     max=max_comps, value=min([10, num_columns]))
                                else:
                                    ui.update_slider("decomposition_max_nc_slider", max=5, value=5)
                                
                                deco_columns = set(col_nbs + discrete_labels(data_in, max_cats=50))
                                if len(columns) < len(deco_columns):
                                    ui.update_checkbox("decomposition_select_all_checkbox", value=False)
                                
                                cat_col = []
                                for c in columns:
                                    if c in col_nums:
                                        nc = len(data_in[to_selected_columns(c, data_in)].unique())
                                        if nc <= 30:
                                            cat_col.append(c)
                                ui.update_selectize("decomposition_numeric_cats_selectize", choices=cat_col)

                            @reactive.effect
                            @reactive.event(input.decomposition_select_all_checkbox, ignore_init=True)
                            def decomposition_columns_update():
                                if input.decomposition_select_all_checkbox():
                                    data_in = node_input.get()["data"]
                                    deco_columns = list(set(col_nbs + discrete_labels(data_in, max_cats=50)))
                                    ui.update_selectize("decomposition_columns_selectize",
                                                        selected=deco_columns)
                            
                            @reactive.effect
                            @reactive.event(input.decomposition_method_selectize)
                            def decomposition_params_ui_update():
                                if input.decomposition_method_selectize() == "KernelPCA":
                                    ui.update_navset("decomposition_params_label", selected="kernelpca_label")
                                    ui.update_navset("decomposition_params_ui", selected="kernelpca_ui")
                                else:
                                    ui.update_navset("decomposition_params_label", selected="empty_label")
                                    ui.update_navset("decomposition_params_ui", selected="empty_ui")

                        elif op_type == "Random sampling":
                            with ui.layout_columns(col_widths=(5, 2, 5, 12), gap="10px"):
                                ui.input_checkbox("randsampling_select_all_checkbox", "Select all")
                                ui.HTML("")
                                ui.input_switch("randsampling_dropna_switch", "Drop NA")
                                ui.input_selectize("randsampling_columns_selectize", "Columns", 
                                                   choices=columns, selected=[],
                                                   multiple=True, remove_button=True)

                            with ui.layout_columns(col_widths=(5, 7, 5, 7, 5, 7, 6, 6), gap="10px"):
                                inline_label("Sample size", pt="22px")
                                num_rows = data_in.shape[0]
                                ui.input_slider("randsampling_size_slider", "",
                                                min=1, max=num_rows, value=num_rows, step=1)
                                inline_label("Batch number")
                                ui.input_numeric("randsampling_batch_numeric", "",
                                                 min=1, max=100, value=1, step=1)
                                inline_label("Random state")
                                ui.input_numeric("randsampling_randstate_numeric", "",
                                                 min=0, max=10000, value=0, step=1)
                                ui.input_switch("randsampling_replace_switch", "Replace")
                                ui.input_switch("randsampling_reset_switch", "Reset index")
                            
                            @reactive.effect
                            @reactive.event(input.randsampling_select_all_checkbox)
                            def randsampling_columns_update():
                                if input.randsampling_select_all_checkbox():
                                    data_in = node_input.get()["data"]
                                    ui.update_selectize("randsampling_columns_selectize",
                                                        selected=to_column_choices(data_in.columns))
                            
                            @reactive.effect
                            @reactive.event(input.randsampling_columns_selectize, ignore_init=True)
                            def randsampling_select_all_update():
                                columns = to_selected_columns(input.randsampling_columns_selectize(), data_in)
                                if len(columns) < data_in.shape[1]:
                                    ui.update_checkbox("randsampling_select_all_checkbox", value=False)
                            
                            @reactive.effect
                            @reactive.event(input.randsampling_dropna_switch)
                            def randsampling_size_slider_update():
                                data_in = node_input.get()["data"]
                                if input.randsampling_dropna_switch():
                                    data_in = data_in.dropna()

                        elif op_type == "Over sampling":
                            ui.input_selectize("over_sampling_target_selectize", "Categorical target",
                                               choices=[""] + col_cats)
                            ui.input_selectize("over_sampling_features_selectize", "Features",
                                               choices=columns,
                                               multiple=True, remove_button=True)
                            ui.input_selectize("over_sampling_method_selectize", "Method",
                                               choices=["Random over-sampling", "SMOTE", "ADASYN"])
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
                                    choices = ["Random over-sampling", "SMOTE", "ADASYN"]
                                else:
                                    choices = ["Random over-sampling", "SMOTE"]
                                ui.update_selectize("over_sampling_method_selectize",
                                                    choices=choices)

                        elif op_type == "Add columns":
                            choices = ["Arithmetic expression",
                                       "Type conversion", "String operations",
                                       "To dummies", "To segments"]
                            ui.input_selectize("add_cols_type_selectize", "Expression type",
                                               choices=choices)
                            
                            label_dict = {"Arithmetic expression": "Formula",
                                          "Type conversion": "Data type",
                                          "String operations": "Methods",
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
                        op_default_name = default_name(var_names.get())
                        ui.input_text("op_name_out_text", "Output name", value=op_default_name,
                                      placeholder=op_default_name)
                        
                        ui.input_text_area("op_markdown_text_area", "Markdown",
                                        placeholder="Key in notes...", height="100px")

                    @reactive.effect
                    @reactive.event(input.add_filter_button, ignore_init=True)
                    def add_filter_button_action():
                        filters = op_memory.get()
                        operator = input.filter_operator_selectize()
                        compared_value = None if operator in ["is True", "not True"] else input.filter_value_text()
                        filters.append(dict(column=input.filter_column_selectize(),
                                            operator=operator,
                                            value=compared_value))
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
                            node = node_input.get()
                            name_in = node["name"]
                            name_out = ops_dict["source"]["name_out"]
                            data_out = ops_dict["data_out"]
                            initial = ops_dict["source"]["code"] == (
                                f"{name_out} = {name_in}.copy()\n"
                                f"{name_out}"
                            )
                            if isinstance(data_out, str) or data_out is None or name_out == "" or initial:
                                ui.update_action_button('save_data_button', disabled=True)
                            else:
                                ui.update_action_button('save_data_button', disabled=False)

                with ui.layout_columns(col_widths=(2, 8, 2)):
                    ui.input_action_button("cancel_data_button", "Cancel", value=0)
                    ui.HTML("")
                    ui.input_action_button("save_data_button", "Save")

                @reactive.effect
                @reactive.event(input.cancel_data_button, input.save_data_button)
                def save_cancel_data_button_action():

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
                    #var_names.set(all_names.copy())
                    op_default_name = default_name(all_names)
                    ui.update_text("op_name_out_text", value=op_default_name, placeholder=op_default_name)

        with ui.nav_panel(None, value="dvs_panel"):
            color = reactive.value('#1f77b4')
            with ui.layout_sidebar(height="900px"):
                with ui.sidebar(bg='#f8f8f8', width='350px', height='900px'):
                
                    @render.express
                    def dvs_panel_ui():
                        node = node_input.get()
                        data = node["data"]
                        if data is None:
                            return

                        dv_type = dv_selected.get()
                        with ui.card(style="background-color:#f8f8f8;border:none;box-shadow:none"):
                            with ui.card_header(style="padding:5px;border:none"):
                                ui.span(f"{dv_type}", style="font-size:16pt")
                                ui.HTML("&nbsp;&nbsp;&nbsp;")
                                with ui.popover(id="dvs_popover", placement="right"):
                                    question_circle_fill
                                    ui.HTML(doc_html(dvs_menu_dict[dv_type]))

                        dvs_dict = dvs.get()
                        with ui.navset_tab(id="visual_config_nav"):
                            with ui.nav_panel("Plot"):

                                columns = to_column_choices(data.columns)
                                col_nums, col_cats, col_nbs = num_cat_labels(data)

                                if dv_type == "Value counts":
                                    choices = [""] + discrete_labels(data, max_cats=100)
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Column")
                                        ui.input_selectize("value_counts_column_selectize", "", choices=choices)

                                    with ui.layout_columns(col_widths=(6, 6, 3, 9, 3, 9), gap="10px"):
                                        ui.input_selectize("value_counts_direction_selectize", "Direction",
                                                           choices=["Vertical", "Horizontal"])
                                        ui.input_selectize("value_counts_method_selectize", "Method",
                                                           choices=["Count", "Density"])

                                        inline_label("Palette", pt="8px")
                                        with ui.layout_columns(col_widths=(5, 7), gap="2px"):
                                            @render.ui
                                            def value_counts_hexcolor():
                                                c = color.get()
                                                return ui.HTML(f"<span style='{hc_style}'>{c}</span>")
                                            ui_color_input("value_counts_color_input", "", value='#1f77b4')
                                        
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("value_counts_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)

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
                                        
                                        @reactive.effect
                                        @reactive.event(input.value_counts_color_input)
                                        def update_value_counts_color():
                                            c = input.value_counts_color_input()
                                            color.set(c)
                                
                                elif dv_type == "Histogram":
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Column")
                                        ui.input_selectize("hist_column_selectize", "",
                                                           choices=[""]+col_nums)

                                        choices = [""] + discrete_labels(data, max_cats=8)
                                        inline_label("Group")
                                        ui.input_selectize("hist_group_by_selectize", "",
                                                           choices=choices, remove_button=True,
                                                           options={"placeholder": "None"})
                                    
                                    with ui.layout_columns(col_widths=(6, 6), gap="10px"):
                                        ui.input_numeric("hist_bins_numeric", "Bins", min=5, max=80, value=10)
                                        ui.input_selectize("hist_method_selectize", "Method",
                                                           choices=["Count", "Density"])
                                    
                                    shift()
                                    with ui.navset_hidden(id="hist_conditional_ui"):
                                        with ui.nav_panel(None, value="hist_single_case"):
                                            with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                                inline_label("Palette", pt="8px")
                                                with ui.layout_columns(col_widths=(5, 7), gap="2px"):
                                                    @render.ui
                                                    def hist_hexcolor():
                                                        c = color.get()
                                                        return ui.HTML(f"<span style='{hc_style}'>{c}</span>")
                                                    ui_color_input("hist_color_input", "", value='#1f77b4')
                                        
                                            @reactive.effect
                                            @reactive.event(input.hist_color_input)
                                            def update_hist_color():
                                                c = input.hist_color_input()
                                                color.set(c)

                                        with ui.nav_panel(None, value="hist_multiple_case"):
                                            with ui.layout_columns(col_widths=(6, 6, 3, 9), gap="10px"):
                                                ui.input_selectize("hist_grouped_norm_selectize", "Normalized",
                                                                choices=["Separately", "Jointly"])
                                                ui.input_selectize("hist_grouped_multiple_selectize", "Style",
                                                                choices=["Layer", "Stack", "Fill"])
                                                inline_label("Theme")
                                                ui.input_selectize("hist_grouped_cmap_selectize", "",
                                                                choices=cat_cmaps, selected="tab10")
                                    
                                    shift()
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("hist_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)
                                    
                                    @reactive.effect
                                    @reactive.event(input.hist_column_selectize,
                                                    input.hist_method_selectize)
                                    def hist_labels_update():

                                        column = input.hist_column_selectize()
                                        method = input.hist_method_selectize()

                                        ui.update_text("fig_xlabel_text", value=column)
                                        ui.update_text("fig_ylabel_text", value=method)
                                    
                                    @reactive.effect
                                    @reactive.event(input.hist_group_by_selectize)
                                    def hist_group_by_selectize_update_ui():

                                        if input.hist_group_by_selectize() == "":
                                            ui.update_navset("hist_conditional_ui", selected="hist_single_case")
                                        else:
                                            ui.update_navset("hist_conditional_ui", selected="hist_multiple_case")
                                
                                elif dv_type == "KDE":
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Column")
                                        ui.input_selectize("kde_column_selectize", "",
                                                           choices=[""]+col_nums)

                                        choices = [""] + discrete_labels(data, max_cats=8)
                                        inline_label("Group")
                                        ui.input_selectize("kde_group_by_selectize", "",
                                                        choices=choices, remove_button=True,
                                                        options={"placeholder": "None"})
    
                                    with ui.navset_hidden(id="kde_conditional_ui"):
                                        with ui.nav_panel(None, value="kde_single_case"):
                                            with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                                inline_label("Palette", pt="8px")
                                                with ui.layout_columns(col_widths=(5, 7), gap="2px"):
                                                    @render.ui
                                                    def kde_hexcolor():
                                                        c = color.get()
                                                        return ui.HTML(f"<span style='{hc_style}'>{c}</span>")
                                                    ui_color_input("kde_color_input", "", value='#1f77b4')
                                        
                                                @reactive.effect
                                                @reactive.event(input.kde_color_input)
                                                def update_kde_color():
                                                    c = input.kde_color_input()
                                                    color.set(c)
                                        
                                        with ui.nav_panel(None, value="kde_multiple_case"):
                                            with ui.layout_columns(col_widths=(6, 6, 3, 9), gap="10px"):
                                                ui.input_selectize("kde_grouped_norm_selectize", "Normalized",
                                                                choices=["Separately", "Jointly"])
                                                ui.input_selectize("kde_grouped_multiple_selectize", "Style",
                                                                choices=["Layer", "Stack", "Fill"])

                                                inline_label("Theme")
                                                ui.input_selectize("kde_grouped_cmap", "",
                                                                choices=cat_cmaps, selected="tab10")
                                    
                                    shift()
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("kde_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)
                                    
                                    @reactive.effect
                                    @reactive.event(input.kde_column_selectize)
                                    def hist_labels_update():

                                        column = input.kde_column_selectize()
                                        
                                        ui.update_text("fig_xlabel_text", value=column)
                                        ui.update_text("fig_ylabel_text", value="Density")
                                    
                                    @reactive.effect
                                    @reactive.event(input.kde_group_by_selectize)
                                    def kde_group_by_selectize_update_ui():

                                        if input.kde_group_by_selectize() == "":
                                            ui.update_navset("kde_conditional_ui", selected="kde_single_case")
                                        else:
                                            ui.update_navset("kde_conditional_ui", selected="kde_multiple_case")

                                elif dv_type == "Box plot":
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
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
                                    
                                    with ui.layout_columns(col_widths=(6, 6), gap="10px"):
                                        ui.input_switch("boxplot_notch_switch", "Notch")
                                        ui.input_switch("boxplot_mean_switch", "Mean")
                                        ui.input_selectize("boxplot_direction_selectize", "Direction",
                                                           choices=["Vertical", "Horizontal"])
                                        ui.input_numeric("boxplot_width_numeric", "Width",
                                                         min=0.1, max=1, step=0.05, value=0.8)
                                    
                                    shift()
                                    with ui.navset_hidden(id="boxplot_conditional_ui"):
                                        with ui.nav_panel(None, value="boxplot_single_case"):
                                            with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                                inline_label("Palette", pt="8px")
                                                with ui.layout_columns(col_widths=(5, 7), gap="2px"):
                                                    @render.ui
                                                    def boxplot_hexcolor():
                                                        c = color.get()
                                                        return ui.HTML(f"<span style='{hc_style}'>{c}</span>")
                                                    ui_color_input("boxplot_color_input", "", value='#1f77b4')
                                        
                                                @reactive.effect
                                                @reactive.event(input.boxplot_color_input)
                                                def update_boxplot_color():
                                                    c = input.boxplot_color_input()
                                                    color.set(c)
                                        
                                        with ui.nav_panel(None, value="boxplot_multiple_case"):
                                            with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                                inline_label("Theme")
                                                ui.input_selectize("boxplot_grouped_cmap_selectize", "",
                                                                   choices=cat_cmaps, selected="tab10")
                                    
                                    shift()
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("boxplot_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)

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
    
                                    @reactive.effect
                                    @reactive.event(input.boxplot_hue_selectize)
                                    def boxplot_group_by_selectize_update_ui():

                                        if input.boxplot_hue_selectize() == "":
                                            ui.update_navset("boxplot_conditional_ui",
                                                             selected="boxplot_single_case")
                                        else:
                                            ui.update_navset("boxplot_conditional_ui",
                                                             selected="boxplot_multiple_case")
                                
                                elif dv_type == "Probability plot":
                                    with ui.layout_columns(col_widths=(3, 9, 12), gap="10px"):
                                        inline_label("Column")
                                        ui.input_selectize("proba_plot_selectize", "", choices=[""] + col_nums)
                                        ui.input_switch("proba_plot_standardize_switch", "Standardize")

                                    with ui.layout_columns(col_widths=(4, 8, 3, 9, 3, 9, 3, 9), gap="10px"):
                                        distr_choices = ["Normal", "Exponential", "Uniform"]
                                        inline_label("Distribution")
                                        ui.input_selectize("proba_plot_distri_selectize", "",
                                                           choices=distr_choices)
                                        inline_label("Palette", pt="8px")
                                        with ui.layout_columns(col_widths=(5, 7), gap="2px"):
                                            @render.ui
                                            def proba_plot_hexcolor():
                                                c = color.get()
                                                return ui.HTML(f"<span style='{hc_style}'>{c}</span>")
                                            ui_color_input("proba_plot_color_input", "", value='#1f77b4')
                                        
                                        @reactive.effect
                                        @reactive.event(input.proba_plot_color_input)
                                        def update_bar_color():
                                            c = input.proba_plot_color_input()
                                            color.set(c)

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
                                    with ui.layout_columns(col_widths=(12, 12), gap="10px"):
                                        ui.input_selectize("pair_columns_selectize", "Columns", 
                                                           choices=[""] + col_nums,
                                                           multiple=True, remove_button=True)
                                        ui.input_selectize("pair_drop_rows_selectize", "Drop rows",
                                                           choices=[""], multiple=True, remove_button=True,
                                                           options={"placeholder": "None"})
                                    
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Hues")
                                        hue_choices = [""] + discrete_labels(data, max_cats=8)
                                        ui.input_selectize("pair_hue_selectize", "",
                                                           choices=[""] + hue_choices, remove_button=True,
                                                           options={"placeholder": "None"})
                                        inline_label("Theme")
                                        ui.input_selectize("pair_cmap_selectize", "",
                                                           choices=cat_cmaps, selected="tab10")
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("pair_alpha_slider", "",
                                                        min=0.2, max=1, step=0.05, value=1)
                                    
                                    with ui.layout_columns(col_widths=(6, 6), gap="10px"):
                                        ui.input_selectize("pair_kind_selectize", "Plot kind", 
                                                           choices=["scatter", "kde", "hist", "reg"])
                                        ui.input_selectize("pair_diag_kind_selectize", "Diagonal kind", 
                                                           choices=["auto", "kde", "hist"])
                                        ui.input_switch("pair_corner_switch", "Corner")
                                
                                    @reactive.effect
                                    def pair_columns_selectize_choices_update():
                                        cols = list(input.pair_columns_selectize())
                                        if len(cols) > 0:
                                            ui.update_selectize("pair_drop_rows_selectize", choices=cols)

                                elif dv_type == "Heat map":
                                    choices = [c for c in col_nbs
                                               if data[to_selected_columns(c, data)].notnull().sum() <= 500]
                                    ui.input_selectize("heatmap_columns_selectize", "Columns",
                                                       choices=choices, selected=[], remove_button=True,
                                                       multiple=True)
                                    
                                    with ui.layout_columns(col_widths=(4, 8, 6, 6), gap="10px"):
                                        inline_label("Theme")
                                        ui.input_selectize("heatmap_colormap_selectize", "",
                                                           choices=num_cmaps)
                                        ui.input_switch("heatmap_annot_switch", "Annotate", value=True)
                                        ui.input_switch("heatmap_top_tick_switch", "Ticks at top", value=True)

                                elif dv_type == "Bar chart":
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Y-data")
                                        choices = [""] + col_nums if data.shape[0] <= 200 else []
                                        ui.input_selectize("bar_ydata_selectize", "", choices=choices)
                                        inline_label("Label")
                                        ui.input_text("bar_label_text", "", placeholder="None")

                                        inline_label("Palette", pt="8px")
                                        with ui.layout_columns(col_widths=(5, 7), gap="2px"):
                                            @render.ui
                                            def bar_hexcolor():
                                                c = color.get()
                                                return ui.HTML(f"<span style='{hc_style}'>{c}</span>")
                                            ui_color_input("bar_color_input", "", value=init_color.get())
                                        
                                        @reactive.effect
                                        @reactive.event(input.bar_color_input)
                                        def update_bar_color():
                                            c = input.bar_color_input()
                                            color.set(c)
                                    
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.HTML("")
                                        ui.input_action_button("bar_add_button", "New bar")

                                        @reactive.effect
                                        @reactive.event(input.bar_ydata_selectize)
                                        def bar_add_button_disable():
                                            ui.update_action_button("bar_add_button",
                                                                    disabled=input.bar_ydata_selectize() == "")
                                    
                                    hr(1, 0.4)
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Sorting")
                                        ui.input_selectize("bar_sort_type_selectize", "",
                                                           choices=["Not applied", "Ascending", "Descending"],
                                                           selected="Not applied")
                                    
                                    with ui.navset_hidden(id="bar_sort_by_conditional_ui"):
                                        with ui.nav_panel(None, value="bar_no_sort_ui"):
                                            ui.HTML("")
                                        with ui.nav_panel(None, value="bar_sort_by_ui"):
                                            shift()
                                            with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                                inline_label("Sort by")
                                                ui.input_selectize("bar_sort_by_selectize", "",
                                                                   choices=[""], remove_button=True,
                                                                   options={"placeholder": "Row index"})
                                    
                                    @reactive.effect
                                    @reactive.event(input.bar_ydata_selectize,
                                                    input.bar_xdata_selectize,
                                                    input.bar_add_button)
                                    def update_bar_sort_by_choices():
                                        choices = []
                                        bars = dv_memory.get().copy()
                                        for bar in bars:
                                            choices.append(bar["ydata"])
                                        if input.bar_ydata_selectize() != "":
                                            choices.append(input.bar_ydata_selectize())
                                        if input.bar_xdata_selectize() != "":
                                            choices.append(input.bar_xdata_selectize())
                                        ui.update_selectize("bar_sort_by_selectize",
                                                            choices=choices, selected="")
                                    
                                    @reactive.effect
                                    @reactive.event(input.bar_sort_type_selectize)
                                    def bar_sort_selectize_update_ui():
                                        if input.bar_sort_type_selectize() in ["Ascending", "Descending"]:
                                            ui.update_navset("bar_sort_by_conditional_ui",
                                                             selected="bar_sort_by_ui")
                                        else:
                                            ui.update_navset("bar_sort_by_conditional_ui",
                                                             selected="bar_no_sort_ui")

                                    shift()
                                    with ui.layout_columns(col_widths=(3, 9, 6, 6, 3, 9, 3, 9), gap="10px"):
                                        inline_label("X-data")
                                        ui.input_selectize("bar_xdata_selectize", "",
                                                           choices=[""]+columns, remove_button=True,
                                                           options={"placeholder": "Row index"})
                                    
                                        dt = ["Vertical", "Horizontal"]
                                        ui.input_selectize("bar_direction_selectize", "Direction", choices=dt)
                                        btype = ["Clustered", "Stacked"]
                                        ui.input_selectize("bar_mode_selectize", "Style", choices=btype)

                                        inline_label("Width", pt="22px")
                                        ui.input_slider("bar_width_slider", "",
                                                        min=0.1, max=1.0, value=0.8, step=0.05)
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("bar_alpha_slider", "",
                                                        min=0.2, max=1.0, value=1.0, step=0.05)
                                    
                                    @reactive.effect
                                    @reactive.event(input.bar_ydata_selectize)
                                    def bar_labels_update():
                                        if input.bar_ydata_selectize() != "":
                                            ui.update_text("bar_label_text",
                                                           placeholder=input.bar_ydata_selectize())
                                
                                elif dv_type == "Radar chart":
                                    with ui.layout_columns(col_widths=(12, 3, 9), gap="10px"):
                                        choices = [""] + col_nums if data.shape[0] <= 200 else []
                                        ui.input_selectize("radar_selectize", "Columns",
                                                           choices=choices, multiple=True, remove_button=True)
                                        inline_label("Category")
                                        ui.input_selectize("radar_cats_selectize", "",
                                                           choices=[""] + columns, remove_button=True,
                                                           options={"placeholder": "Row index"})
                                    
                                    with ui.layout_columns(col_widths=(5, 7), gap="10px"):
                                        inline_label("Tick axis angle", pt="22px")
                                        ui.input_slider("radar_tick_angle_slider", "",
                                                        min=0, max=355, value=0, step=5)
                                    
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Theme")
                                        ui.input_selectize("radar_cmap_selectize", "",
                                                           choices=cat_cmaps, selected="tab10")
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("radar_alpha_slider", "",
                                                        min=0.2, max=1.0, value=0.6, step=0.05)

                                elif dv_type == "Line plot":
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
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
                                        inline_label("Label")
                                        ui.input_text("line_label_text", "", placeholder="None")

                                        @reactive.effect
                                        @reactive.event(input.line_ydata_selectize)
                                        def line_labels_update():
                                            if input.line_ydata_selectize() != "":
                                                ui.update_text("line_label_text",
                                                               placeholder=input.line_ydata_selectize())
                                                
                                    with ui.layout_columns(col_widths=(6, 6, 3, 9, 3, 9, 3, 9), gap="10px"):
                                        styles = ["solid", "dash", "dot", "dash-dot"]
                                        ui.input_selectize("line_style_selectize", "Style", choices=styles)
                                        markers = ["none", "circle", "square", "dot",
                                                   "diamond", "triangle", "star", "cross"]
                                        ui.input_selectize("line_marker_selectize", "Marker", choices=markers)
                                    
                                        inline_label('Width', pt="22px")
                                        ui.input_slider("line_width_slider", "",
                                                        min=0.5, max=4, step=0.5, value=1.5)

                                        inline_label("Scale", "22px")
                                        ui.input_slider("line_marker_scale_slider", "",
                                                        min=0.1, max=2, step=0.05, value=1)

                                        inline_label("Palette", pt="8px")
                                        with ui.layout_columns(col_widths=(5, 7), gap="2px"):
                                            @render.ui
                                            def line_hexcolor():
                                                c = color.get()
                                                return ui.HTML(f"<span style='{hc_style}'>{c}</span>")
                                            ui_color_input("line_color_input", "", value=init_color.get())
                                        
                                        @reactive.effect
                                        @reactive.event(input.line_color_input)
                                        def update_line_color():
                                            c = input.line_color_input()
                                            color.set(c)
                                    
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.HTML("")
                                        ui.input_action_button("line_add_button", "New line", )
                                    
                                    @reactive.effect
                                    @reactive.event(input.line_ydata_selectize)
                                    def line_add_button_disable():

                                        ui.update_action_button("line_add_button",
                                                                disabled=input.line_ydata_selectize() == "")
                                
                                elif dv_type == "Scatter plot":
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
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

                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Size")
                                        ui.input_selectize("scatter_size_data_selectize", "",
                                                           choices=[""]+col_nums,
                                                           remove_button=True, options={"placeholder": "None"})
                                        inline_label('Scale', pt="22px")
                                        ui.input_slider("scatter_size_scale_slider", "",
                                                        min=0.1, max=2, value=1, step=0.05)
                                    
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Hues")
                                        ui.input_selectize("scatter_color_data_selectize", "",
                                                           choices=[""]+columns,
                                                           remove_button=True, options={"placeholder": "None"})
                                    
                                    shift()
                                    with ui.navset_hidden(id="scatter_conditional_ui"):
                                        with ui.nav_panel(None, value="scatter_single_case"):
                                            with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                                inline_label("Palette", pt="8px")
                                                with ui.layout_columns(col_widths=(5, 7), gap="2px"):
                                                    @render.ui
                                                    def scatter_hexcolor():
                                                        c = color.get()
                                                        return ui.HTML(f"<span style='{hc_style}'>{c}</span>")
                                                    ui_color_input("scatter_color_input", "", value='#1f77b4')
                                        
                                            @reactive.effect
                                            @reactive.event(input.scatter_color_input)
                                            def update_scatter_color():
                                                c = input.scatter_color_input()
                                                color.set(c)
                                        
                                        with ui.nav_panel(None, value="scatter_multiple_case"):
                                            with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                                inline_label("Theme")
                                                ui.input_selectize("scatter_cmap_selectize", "",
                                                                   choices=num_cmaps, selected="viridis")
                                    
                                    shift()
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("scatter_alpha_slider", "",
                                                        min=0.2, max=1.0, value=1.0, step=0.05)
                                    
                                    @reactive.effect
                                    @reactive.event(input.scatter_color_data_selectize)
                                    def scatter_color_data_selectize_update_ui():

                                        color_data = input.scatter_color_data_selectize()

                                        if color_data == "":
                                            ui.update_navset("scatter_conditional_ui",
                                                             selected="scatter_single_case")
                                        else:
                                            ui.update_navset("scatter_conditional_ui",
                                                             selected="scatter_multiple_case")
                                            
                                            if color_data in col_cats:
                                                cmaps, cmap = cat_cmaps, "tab10"
                                            else:
                                                cmaps, cmap = num_cmaps, "viridis"
                                            ui.update_selectize("scatter_cmap_selectize",
                                                                choices=cmaps, selected=cmap)
                                
                                elif dv_type == "Regression plot":
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
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
                                    
                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        choices = [""] + discrete_labels(data, max_cats=8)
                                        inline_label("Hues")
                                        ui.input_selectize("regplot_color_data_selectize", "",
                                                           choices=choices,
                                                           remove_button=True, options={"placeholder": "None"})
                                    shift()
                                    with ui.navset_hidden(id="regplot_conditional_ui"):
                                        with ui.nav_panel(None, value="regplot_single_case"):
                                            with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                                inline_label("Palette", pt="8px")
                                                with ui.layout_columns(col_widths=(5, 7), gap="2px"):
                                                    @render.ui
                                                    def regplot_hexcolor():
                                                        c = color.get()
                                                        return ui.HTML(f"<span style='{hc_style}'>{c}</span>")
                                                    ui_color_input("regplot_color_input", "", value='#1f77b4')
                                        
                                                @reactive.effect
                                                @reactive.event(input.regplot_color_input)
                                                def update_regplot_color():
                                                    c = input.regplot_color_input()
                                                    color.set(c)
                                        
                                        with ui.nav_panel(None, value="regplot_multiple_case"):
                                            with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                                inline_label("Theme")
                                                ui.input_selectize("regplot_cmap_selectize", "",
                                                                   choices=num_cmaps, selected="viridis")
                                    shift()
                                    with ui.layout_columns(col_widths=(3, 9)):
                                        inline_label("Opacity", pt="22px")
                                        ui.input_slider("regplot_alpha_slider", "",
                                                        min=0.2, max=1.0, value=1.0, step=0.05)
                                    
                                    with ui.layout_columns(col_widths=(6, 6), gap="10px"):
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
                                            shift()
                                            with ui.layout_columns(col_widths=(6, 6), gap="10px"):
                                                inline_label("Polynomial order")
                                                ui.input_numeric("regplot_poly_order_numeric", "", 
                                                                 min=2, max=10, step=1, value=2)

                                    @reactive.effect
                                    @reactive.event(input.regplot_ydata_selectize,
                                                    input.regplot_xdata_selectize)
                                    def regplot_transform_choices_update():
                                        node = node_input.get()
                                        data = node["data"]
                                        ylabel = to_selected_columns(input.regplot_ydata_selectize(), data)
                                        xlabel = to_selected_columns(input.regplot_xdata_selectize(), data)
                                        if xlabel != "" and ylabel != "":
                                            choices = ["None", "Polynomial"]
                                            xvalue = data[xlabel]
                                            if (xvalue > 0).all():
                                                choices.append("Log")
                                            yvalue = data[ylabel]
                                            if is_bool_dtype(yvalue) or ((yvalue*(1-yvalue) >= 0).all()):
                                                choices.append("Logistic")
                                            ui.update_selectize("regplot_transform_selectize",
                                                                choices=choices)

                                    @reactive.effect
                                    @reactive.event(input.regplot_color_data_selectize)
                                    def regplot_color_data_selectize_update_ui():

                                        color_data = input.regplot_color_data_selectize()

                                        if color_data == "":
                                            ui.update_navset("regplot_conditional_ui",
                                                             selected="regplot_single_case")
                                        else:
                                            ui.update_navset("regplot_conditional_ui",
                                                             selected="regplot_multiple_case")
                                            cmaps, cmap = cat_cmaps, "tab10"
                                            ui.update_selectize("regplot_cmap_selectize",
                                                                choices=cmaps, selected=cmap)

                                elif dv_type == "Filled areas":

                                    with ui.layout_columns(col_widths=(12, 3, 9), gap="10px"):
                                        ui.input_selectize("filled_areas_ydata_selectize", "Y-data",
                                                           choices=[""] + col_nums,
                                                           multiple=True, remove_button=True)
                                        inline_label("X-data")
                                        ui.input_selectize("filled_areas_xdata_selectize", "",
                                                           choices=[""] + columns, remove_button=True,
                                                           options={"placeholder": "Row index"})

                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Style")
                                        ui.input_selectize("filled_areas_style_selectize", "",
                                                           choices=["Layer", "Stack"], selected="Stack")
                                        inline_label("Theme")
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
                                    with ui.layout_columns(col_widths=(6, 6), gap="10px"):
                                        ui.input_selectize("ac_plot_type_selectize", "Plot type",
                                                           choices=["ACF", "PACF"])
                                        ui.input_selectize("ac_plot_method_selectize", "Method", choices=[""])
                                        ui.input_numeric("ac_plot_lags_numeric", "Lags",
                                                         min=4, max=max_lags, value=27, step=1)
                                        ui.input_selectize("ac_plot_ci_selectize", "Confidence level",
                                                           choices=["80%", "85%", "90%", "95%", "99%"],
                                                           selected="95%")

                                    with ui.layout_columns(col_widths=(3, 9), gap="10px"):
                                        inline_label("Palette", pt="8px")
                                        with ui.layout_columns(col_widths=(5, 7), gap="2px"):
                                            @render.ui
                                            def ac_plot_hexcolor():
                                                c = color.get()
                                                return ui.HTML(f"<span style='{hc_style}'>{c}</span>")
                                            ui_color_input("ac_plot_color_input", "", value='#1f77b4')
                                        
                                        @reactive.effect
                                        @reactive.event(input.ac_plot_color_input)
                                        def update_ac_plot_color():
                                            c = input.ac_plot_color_input()
                                            color.set(c)

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
                                    with ui.layout_columns(col_widths=(8, 4), gap="10px"):
                                        #label_shift = 
                                        ui.input_text("fig_title_text", "Title")
                                        ui.input_selectize("fig_title_size_selectize",
                                                           inline_label(" ", pt="2px"),
                                                           choices=[f"{i}pt" for i in range(6, 21)],
                                                           selected="12pt", )
                                        
                                        ui.input_text("fig_xlabel_text", "X-label")
                                        ui.input_selectize("fig_xlabel_size_selectize",
                                                           inline_label(" ", pt="2px"),
                                                           choices=[f"{i}pt" for i in range(6, 21)],
                                                           selected="10pt")
                                        
                                        ui.input_text("fig_ylabel_text", "Y-label")
                                        ui.input_selectize("fig_ylabel_size_selectize",
                                                           inline_label(" ", pt="2px"),
                                                           choices=[f"{i}pt" for i in range(6, 21)],
                                                           selected="10pt")

                                        locs = ["upper left", "upper center", "upper right", 
                                                "center left", "center right", "center",
                                                "lower left", "lower center", "lower right"]
                                        ui.input_selectize("fig_legend_loc_selectize",
                                                           "Legend", choices=locs)
                                        ui.input_selectize("fig_legend_size_selectize",
                                                           inline_label(" ", pt="2px"),
                                                           choices=[f"{i}pt" for i in range(6, 21)],
                                                           selected="10pt")
        
                                    with ui.layout_columns(col_widths=(5, 7), gap="10px"):
                                        inline_label("Rotate X-ticks", pt="22px")
                                        ui.input_slider("fig_xtick_rotate_numeric", "",
                                                         min=-90, max=90, step=10, value=0)

                            with ui.nav_panel("Figure"):
                                with ui.layout_columns(col_widths=(5, 2, 5), gap="10px"):
                                    ui.input_switch("fig_grid_switch", "Grid")
                                    ui.HTML("")
                                    if dv_type not in ["Pair plot", "Radar chart", "ACF and PACF"]:
                                        ui.input_switch("fig_equal_axis_switch", "Equal axis")
                                    else:
                                        ui.HTML("")
                                with ui.layout_columns(col_widths=(12, 3, 9, 3, 9), gap="10px"):
                                    ui.markdown("Figure size")
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
                        bars.append(dict(ydata=input.bar_ydata_selectize(),
                                         label=input.bar_label_text().strip(),
                                         color=color.get()))
                        ui.update_selectize("bar_ydata_selectize", selected="")
                        ui.update_text("bar_label_text", value="")

                        index = default_colors.index(init_color.get())
                        init_color.set(default_colors[(index + 1) % len(default_colors)])
                        color.set(init_color.get())
                    
                    @reactive.effect
                    @reactive.event(input.line_add_button, ignore_init=True)
                    def line_add_button_action():
                        
                        lines = dv_memory.get()
                        lines.append(dict(ydata=input.line_ydata_selectize(),
                                          label=input.line_label_text().strip(),
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

                        ui.update_text("line_label_text", value="")

                        index = default_colors.index(init_color.get())
                        init_color.set(default_colors[(index + 1) % len(default_colors)])
                        color.set(init_color.get())
                    
                    #ui.input_action_button("dv_debug", "Debug")
                    #@render.code
                    #@reactive.event(input.dv_debug)
                    #def dv_debug_display():
                    #    return str(dvs.get())

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

                with ui.layout_columns(col_widths=(2, 8, 2)):
                    ui.input_action_button("cancel_visual_button", "Cancel")
                    ui.HTML("")
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

                        with ui.card(style="background-color:#f8f8f8;border:none;box-shadow:none"):
                            with ui.card_header(style="padding:5px;border:none"):
                                ui.span(f"{md_type}", style="font-size:16pt")
                                ui.HTML("&nbsp;&nbsp;&nbsp;")
                                with ui.popover(id="mds_popover", placement="right"):
                                    question_circle_fill
                                    ui.HTML(doc_html(mds_menu_dict[md_type]))

                        with ui.navset_hidden(id="model_page_navset"):
                            with ui.nav_panel("model_page1"):
                                if md_type == "Scikit-learn models":
                                    # ui.markdown("**Step 1: specify variables**")
                                    with ui.card_header(style="padding-bottom:15px;border:none"):
                                        ui.span(ui.HTML("<b>Step 1: specify variables</b>"),
                                                style="font-size:12pt")
                                        ui.HTML("&nbsp;&nbsp;")
                                        with ui.popover(id="sklearn_popover1", placement="right"):
                                            question_circle_fill
                                            ui.HTML(doc_html(sklearn_page_dict[0]))

                                ui.input_selectize("model_dependent_selectize", "Dependent variable",
                                                   choices=[""] + col_predicted)
                                ui.input_selectize("model_independent_selectize", "Independent variables",
                                                   choices=[""] + col_predictors,
                                                   multiple=True, remove_button=True)
                                ui.input_selectize("model_numeric_cats_selectize", "Numbers treated as categories",
                                                   choices=[""], multiple=True, remove_button=True,
                                                   options={"placeholder": "None"})
                                
                                if md_type == "Scikit-learn models":
                                    with ui.layout_columns(col_widths=(6, 6)):
                                        ui.input_switch("model_formula_switch", "Edit formula")

                                        @render.express
                                        def model_drop_first_ui():
                                            if not input.model_formula_switch():
                                                ui.input_switch("model_drop_first_switch","Drop first",
                                                                value=True)

                                with ui.navset_hidden(id="model_formula_ui_navset"):
                                    with ui.nav_panel(None, value="model_formula_off"):
                                        None
                                    with ui.nav_panel(None, value="model_formula_on"):
                                        ui.input_text("statsmodels_formula_text", "Formula")

                                
                                @reactive.effect
                                @reactive.event(input.statsmodels_formula_text)
                                def statsmodels_formula_ui_navset_update():
                                    if md_selected.get() == "Statsmodels":
                                        ui.update_navset("model_formula_ui_navset", selected="model_formula_on")

                                @reactive.effect
                                @reactive.event(input.model_formula_switch, ignore_init=True)
                                def sklearn_formula_ui_navset_update():
                                    if input.model_formula_switch() or md_selected.get() == "Statsmodels":
                                        ui.update_navset("model_formula_ui_navset", selected="model_formula_on")
                                    else:
                                        ui.update_navset("model_formula_ui_navset", selected="model_formula_off")

                                if md_type == "Statsmodels":
                                    ui.input_selectize("statsmodels_type_selectize", "Model type",
                                                       choices=["ols", "logit"])                                    
                                    hr()
                                    sm_default_name = default_name(var_names.get())
                                    ui.input_text("statsmodels_output_text", "Output name",
                                                  value=sm_default_name,
                                                  placeholder=sm_default_name)

                                    @render.express
                                    def fit_statsmodels_ui():
                                        if input.statsmodels_formula_text().strip() != "":
                                            ui.input_task_button("statsmodels_fitting_button", "Fit model",
                                                                 label_busy="Running...", width="100%")

                            with ui.nav_panel("model_page2"):
                                with ui.card_header(style="padding-bottom:15px;border:none"):
                                    ui.span(ui.HTML("<b>Step 2: model pipeline</b>"),
                                            style="font-size:12pt")
                                    ui.HTML("&nbsp;&nbsp;")
                                    with ui.popover(id="sklearn_popover2", placement="right"):
                                        question_circle_fill
                                        ui.HTML(doc_html(sklearn_page_dict[1]))
                                
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
                                        ui.input_selectize("sklearn_over_sampling_selectize",
                                                           "Over-sampling of response",
                                                           choices=os_choices)
                                    else:
                                        ui.input_switch("sklearn_predicted_log_switch",
                                                        "Log transformation of response")

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

                                with ui.layout_columns(col_widths=(4, 8), gap="10px"):
                                    inline_label("Scalilng")
                                    ui.input_selectize("sklearn_scaling_selectize", "",
                                                       choices=["Not applied", "StandardScaler", "Normalizer"])
                                    inline_label("PCA")
                                    ui.input_text("sklearn_pca_numbers", "", placeholder="Not applied")

                                hr()
                                ui.input_selectize("sklearn_model_selectize", "Model selection",
                                                   choices=[""])
                                
                                @reactive.effect
                                @reactive.event(input.sklearn_model_selectize)
                                def sklearn_default_scaling_update():
                                    model_name = input.sklearn_model_selectize()
                                    if model_name in ["LogisticRegression", "KNeighborsRegressor",
                                                      "Lasso", "Ridge"]:
                                        ui.update_selectize("sklearn_scaling_selectize",
                                                            selected="StandardScaler")
                                    else:
                                        ui.update_selectize("sklearn_scaling_selectize",
                                                            selected="Not applied")
                                
                                @render.express
                                def sklearn_model_hypers_ui():
                                    model_name = input.sklearn_model_selectize()
                                    if model_name == "":
                                        return
                                    hyper_list = model_hypers[model_name]
                                    
                                    if len(hyper_list) == 1:
                                        hyper, label_string, default_value, param_doc = hyper_list[0]
                                        with ui.card_header(style="padding-bottom:10px;border:none"):
                                            ui.HTML(f"{label_string} &nbsp;")
                                            with ui.popover(id=f"sklearn_{model_name.lower()}_{hyper}_popover",
                                                            placement="right"):
                                                question_circle_fill
                                                ui.HTML(param_doc)
                                        ui.input_text(f"sklearn_{model_name.lower()}_{hyper}", "",
                                                      placeholder=default_value)
                                    elif len(hyper_list) > 1:
                                        with ui.layout_columns(col_widths=(7, 5), gap="10px"):
                                            for hyper, label_string, default_value, param_doc in hyper_list:
                                                with ui.card_header(style="padding-top:6px;border:none"):
                                                    ui.HTML(f"<span>{label_string} &nbsp;</span>")
                                                    with ui.popover(id=f"sklearn_{model_name.lower()}_{hyper}_popover",
                                                                    placement="right"):
                                                        question_circle_fill
                                                        ui.HTML(param_doc)
                                                #inline_label(label_string)
                                                ui.input_text(f"sklearn_{model_name.lower()}_{hyper}", "",
                                                              placeholder=default_value)
                    
                            with ui.nav_panel("model_page3"):
                                #ui.markdown("**Step 3: model fitting and evaluation**")
                                with ui.card_header(style="padding-bottom:15px;border:none"):
                                    ui.span(ui.HTML("<b>Step 3: model fitting and testing</b>"),
                                            style="font-size:12pt")
                                    ui.HTML("&nbsp;&nbsp;")
                                    with ui.popover(id="sklearn_popover3", placement="right"):
                                        question_circle_fill
                                        ui.HTML(doc_html(sklearn_page_dict[2]))

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
                                #ui.markdown("**Step 4: save results**")
                                with ui.card_header(style="padding-bottom:15px;border:none"):
                                    ui.span(ui.HTML("<b>Step 4: output results</b>"),
                                            style="font-size:12pt")
                                    ui.HTML("&nbsp;&nbsp;")
                                    with ui.popover(id="sklearn_popover4", placement="right"):
                                        question_circle_fill
                                        ui.HTML(doc_html(sklearn_page_dict[3]))
                                
                                ui.input_checkbox_group("sklearn_outputs_checkbox",
                                                        inline_label("Output figures", pt="10px"),
                                                        choices=[])

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
                                
                                @render.express
                                def sklearn_feature_importance_number_ui():
                                    model_name = input.sklearn_model_selectize()
                                    if "Tree" in model_name or "Forest" in model_name or "Boosting" in model_name:
                                        if "Feature importance" in input.sklearn_outputs_checkbox():
                                            if "model" not in mds_dict["memory"]:
                                                return
                                            model_step = mds_dict["memory"]["model"][-1]
                                            if mds_dict["type"] == "Regressor":
                                                if input.sklearn_predicted_log_switch():
                                                    model_step = model_step.regressor_
                                            importances = model_step.feature_importances_
                                            max_features = len(importances)
                                            min_features = min(5, max_features)
                                            with ui.layout_columns(col_widths=(4, 8), gap="10px"):
                                                inline_label("Features no.", pt="22px")
                                                ui.input_slider("sklearn_feature_importance_number_slider", "",
                                                                min=min_features, max=max_features,
                                                                value=min(10, max_features), step=1)

                                hr()
                                sl_default_name = default_name(var_names.get())
                                ui.input_text("sklearn_output_text", "Output name",
                                              value=sl_default_name,
                                              placeholder=sl_default_name)
                                
                                @render.express
                                def sklearn_regression_out_ui():
                                    mds_dict = mds.get()
                                    if mds_dict["type"] == "Regressor":
                                        ui.input_switch("sklearn_residual_switch", "Include residuals")

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
                                    
                                    model_name = input.sklearn_model_selectize()
                                    if "Tree" in model_name or "Forest" in model_name or "Boosting" in model_name:
                                        choices.append("Feature importance")
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

                        if md_type == "Scikit-learn models":
                            with ui.layout_columns(col_widths=(5, 2, 5)):
                                ui.input_action_button("sklearn_page_back_button", "Back", disabled=True)
                                ui.HTML("")
                                ui.input_action_button("sklearn_page_next_button", "Next", disabled=True)

                    #ui.input_action_button("md_debug", "Debug")
                    #@render.code
                    #@reactive.event(input.md_debug)
                    #def md_debug_display():
                    #    return str(md_memory.get())

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
                                    input.model_numeric_cats_selectize,
                                    ignore_init=True)
                    def statsmodels_formula_text_update():
                        dependent = input.model_dependent_selectize()
                        independents = input.model_independent_selectize()
                        num_cats = input.model_numeric_cats_selectize()
                        independents = [f"C({c})" if c in num_cats else c for c in independents]

                        if independents != "":
                            if md_selected.get() == "Statsmodels":
                                if dependent != "":
                                    formula = f"{dependent} ~ {' + '.join(independents)}"
                                    ui.update_text("statsmodels_formula_text", value=formula)
                            else:
                                formula = f"{' + '.join(independents)}"
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
                        ui.update_navset("model_page_navset",
                                         selected=f"model_page{md_page.get()}")
                        ui.update_action_button("sklearn_page_back_button", disabled=page < 2)

                        if page == 1:
                            predicted = input.model_dependent_selectize()
                            predictors = input.model_independent_selectize()
                            formula = input.statsmodels_formula_text().strip() 
                            formula_switch = input.model_formula_switch()

                            memory = mds.get()["memory"]
                            if formula_switch and "formua_err" in memory:
                                disabled = predicted == "" or formula == "" or memory["formula_err"] is not None    
                            else:
                                disabled = predicted == "" or len(predictors) == 0

                        elif page == 2:
                            disabled = input.sklearn_model_selectize() == ""
                            md_memory.set({})
                        elif page == 3:
                            disabled = len(md_memory.get()) == 0
                            if not disabled:
                                disabled = not isinstance(md_memory.get()["result"], str)
                        else:
                            disabled = True
                        ui.update_action_button("sklearn_page_next_button", disabled=disabled)

                    @reactive.effect
                    def sklearn_model_choices():

                        mds_dict = mds.get()
                        if md_page.get() >= 2:
                            if mds_dict["type"] == "Regressor":
                                models = ["LinearRegression", "Ridge", "Lasso", "KNeighborsRegressor",
                                          "DecisionTreeRegressor", "RandomForestRegressor",
                                          "GradientBoostingRegressor"]
                                output_choices = ["Prediction plot", "Residual plot"]
                            elif mds_dict["type"] == "Classifier":
                                models = ["LogisticRegression", "KNeighborsClassifier",
                                          "DecisionTreeClassifier", "RandomForestClassifier",
                                          "GradientBoostingClassifier"]
                                output_choices = ["Confusion matrix"]
                            else:
                                models = ["No available model"]
                                output_choices = []
                            
                            selected = input.sklearn_model_selectize()
                            if selected not in models:
                                selected = models[0]

                            if md_page.get() >= 3:
                                if "Tree" in selected or "Forest" in selected or "Boosting" in selected:
                                    output_choices.append("Feature importance")

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
                        summary_lines = result_summary.split("\n")
                        if "OLS" in summary_lines[0] or "Logit" in summary_lines[0]:
                            summary_preview = result_summary
                        else:
                            summary_preview = "No result."
                        name_save = input.statsmodels_output_text().strip()
                        if name_save != "":
                            invalid = invalid_name(name_save, error=True)
                            if invalid is not False:
                                raise invalid
                        mds_dict["memory"] = dict(result=eval("result", statsmodels_ns))
                    except Exception as err:
                        result_summary = err
                        summary_preview = err
                        mds_dict["memory"] = {}
                    mds_dict["results"] = summary_preview
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
                            mds_dict["memory"]["estimator"] = eval("search", sklearn_ns).best_estimator_
                        else:
                            params_code = ""
                            mds_dict["memory"]["estimator"] = eval("model", sklearn_ns)

                        train_result = f"\n\nTraining score: {eval('train_score', sklearn_ns):.4f}"
                        if test_set:
                            test_result = f"\nTest score: {eval('test_score', sklearn_ns):.4f}"
                        else:
                            test_result = ""

                        result = (
                            f"{params_code}"
                            f"{eval('table', sklearn_ns)}\n\n"
                            f"Cross-validation score: {eval('score', sklearn_ns).mean():.4f}"
                            f"{train_result}"
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
                        if name_save == "" or md_page.get() < 4:
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
                        elif md_type == "Scikit-learn models":
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
                                current_imports = mds_dict["source"]["imports"][1]
                                current_code = mds_dict["source"]["code"][1]
                                sklearn_ns = {}
                                if len(current_imports) > 0:
                                    exec('\n'.join(current_imports), sklearn_ns)

                                if current_code == "":
                                    return
                                else:
                                    name = node["name"]
                                    sklearn_ns[name] = data
                                    mds_dict["memory"]["formula_err"] = None
                                    try:
                                        exec(current_code, sklearn_ns)
                                        mds_dict["memory"]["x"] = eval("x", sklearn_ns)
                                        mds_dict["memory"]["y"] = eval("y", sklearn_ns)
                                        if "to_dummies = " in current_code:
                                            mds_dict["memory"]["to_dummies"] = eval("to_dummies", sklearn_ns)
                                    except Exception as err:
                                        mds_dict["memory"]["formula_err"] = err
                                
                                if mds_dict["memory"]["formula_err"] is None:
                                    table_width = len(data.__repr__().split('\n')[0]) * 72 // 96
                                    with ui.layout_column_wrap(width=f"{table_width}px",
                                                               fixed_width=True, fill=False, fillable=False):
                                        @render.table()
                                        def model_data_preview():
                                            predicted = input.model_dependent_selectize()
                                            predictors = list(input.model_independent_selectize())
                                            var_columns = [predicted] + predictors if predicted != "" else predictors
                                            clean_data = data
                                            if len(var_columns) > 0:
                                                if data[to_selected_columns(var_columns, data)].isnull().values.any():
                                                    clean_data = data.dropna(subset=var_columns)
                                            else:
                                                clean_data = data
                                            row, col = clean_data.shape
                                            table = display_table(clean_data, 16).style.format(precision=4)
                                            table.set_caption(f"{row} rows x {col} columns")
                                        
                                            styles = table_styles.copy()
                                            if predicted != "":
                                                col_index = columns.index(predicted)
                                                c = "#ffe7e7"
                                                styles.append(dict(selector=f"td.col{col_index}",
                                                                   props=[("background-color", c)]))
                                            c = "#eae7ff"
                                            for p in predictors:
                                                col_index = columns.index(p)
                                                styles.append(dict(selector=f"td.col{col_index}",
                                                                   props=[("background-color", c)]))

                                            return table.set_table_styles(styles)
                                else:
                                    @render.ui
                                    def sklearn_formula_error():
                                        return ui_block(str(mds_dict["memory"]["formula_err"]), "danger")

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
                                    num_plots = sum([1 for out in outputs if out["type"] == 'plot'])
                                    if "result" in memory and num_plots > 0:
                                        if isinstance(memory["result"], str):
                                            @expressify
                                            def sklearn_plot_display(idx):
                                                sklearn_ns = dict(mds=mds, render=render)
                                                outputs = mds.get()["outputs"]
                                                width, height = outputs[idx]["fig"].get_size_inches() * 100
                                                exec(
                                                    f"@render.plot(width={int(width)}, height={int(height)})\n"
                                                    f"def plot_display_fun{idx}():\n"
                                                    "    mds_dict = mds.get()\n"
                                                    f"    if {idx} < len(mds_dict['outputs']):\n"
                                                    f"        return mds_dict['outputs'][{idx}]['fig']",
                                                    sklearn_ns
                                                )
                                                with ui.card(height=f"{height + 20}px",
                                                             max_height=f"{height + 20}px",
                                                             style="box-shadow:none;box-border:none,padding:0px"):
                                                    output_plot(f"plot_display_fun{idx}",
                                                                width=width, height=height)

                                            with ui.layout_columns(col_widths=(6, 6), gap="5px"):
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
                                                            sklearn_ns[key] = value
                                                        
                                                        exec("\n".join(current_code.split("\n")[:-1]), sklearn_ns)
                                                        out["fig"] = eval("fig", sklearn_ns)
                                                        sklearn_plot_display(idx)
                                                
                                                empty_slots = (-num_plots) % 2
                                                for _ in range(empty_slots):
                                                    ui.HTML("")
                                    
                                    else:
                                        ui.markdown(" ")
                                                        
                with ui.layout_columns(col_widths=(2, 8, 2)):
                    ui.input_action_button("cancel_model_button", "Cancel")
                    ui.HTML("")
                    ui.input_action_button("save_model_button", "Save", disabled=True)
                
                @reactive.effect
                @reactive.event(input.statsmodels_fitting_button, input.statsmodels_output_text)
                def save_statsmodels_button_disable():
                    
                    mds_dict = mds.get()
                    memory = md_memory.get()
                    fit = len(mds_dict["memory"]) > 0 and len(memory) > 0
                    invalid = invalid_name(input.statsmodels_output_text().strip())
                    disabled = invalid or (not fit)
                    ui.update_action_button("save_model_button", disabled=disabled)
                
                @reactive.effect
                @reactive.event(input.sklearn_fitting_button, input.sklearn_output_text,
                                input.sklearn_test_set_switch, input.sklearn_page_next_button)
                def save_sklearn_button_disable():
                    
                    memory = md_memory.get()
                    disabled = True
                    if "result" in memory:
                        invalid = invalid_name(input.sklearn_output_text().strip())
                        disabled = (not isinstance(memory["result"], str)) or (md_page.get() < 4) or invalid
                    
                    ui.update_action_button("save_model_button", disabled=disabled)

                #@reactive.effect
                #@reactive.event(input.cancel_model_button)
                #def cancel_model_to_canvas_action():
                #    node_view.set(None)

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
                                          results=mds_dict["results"],
                                          estimator=mds_dict["memory"]["estimator"])
                        
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
                            width, height = fig.get_size_inches() * 100
                            view = dict(fig=fig, width=width, height=height)
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

            ui.update_navset("main", selected="canvas_panel")
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

            