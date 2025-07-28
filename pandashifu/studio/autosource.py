import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.api.types import is_numeric_dtype, is_bool_dtype
from collections.abc import Iterable
from numbers import Real


model_hypers = {
    "LinearRegression": [],
    "Ridge": [("alpha", "Shrinkage parameter alpha", "1.0")],
    "Lasso": [("alpha", "Shrinkage parameter alpha", "1.0")],
    "DecisionTreeRegressor": [
        ("max_depth", "Max tree depth", "None"),
        ("min_samples_split", "Min samples split", "2"),
        ("min_samples_leaf", "Min samples leaf", "1"),
        ("max_leaf_nodes", "Max leaf nodes", "None")
    ],
    "RandomForestRegressor": [
        ("max_depth", "Max tree depth", "None"),
        ("min_samples_split", "Min samples split", "2"),
        ("min_samples_leaf", "Min samples leaf", "1"),
        ("max_leaf_nodes", "Max leaf nodes", "None"),
        ("max_features", "Max features", "1.0")
    ],
    "LogisticRegression": [("C", "Inverse regularization C", "1.0")],
    "DecisionTreeClassifier": [
        ("max_depth", "Max tree depth", "None"),
        ("min_samples_split", "Min samples split", "2"),
        ("min_samples_leaf", "Min samples leaf", "1"),
        ("max_leaf_nodes", "Max leaf nodes", "None")
    ],
    "RandomForestClassifier": [
        ("max_depth", "Max tree depth", "None"),
        ("min_samples_split", "Min samples split", "2"),
        ("min_samples_leaf", "Min samples leaf", "1"),
        ("max_leaf_nodes", "Max leaf nodes", "None"),
        ("max_features", "Max features", "sqrt")
    ],
}


def to_column_choices(columns):

    if isinstance(columns, pd.MultiIndex):
        return [col.__repr__() for col in columns]
    else:
        return columns.tolist()


def to_selected_columns(columns, data):

    if isinstance(data.columns, pd.MultiIndex):
        if isinstance(columns, str):
            return "" if columns == "" else eval(columns) 
        else:
            return [eval(col) for col in columns]
    else:
        if isinstance(columns, str):
            return columns
        else:
            return list(columns)


def str_to_values(string, sup=False):

    try:
        values = eval(string.strip())
    except Exception as err:
        if sup:
            values = None
        else:
            values = err
    
    return values


def str_to_numstr(string, sup=False):

    if string.strip() == "":
        return None

    try:
        nums = eval(string.strip())
        if isinstance(nums, np.ndarray):
            numstr = string.strip()
        elif isinstance(nums, Iterable):
            numstr = list(nums).__repr__()
        elif isinstance(nums, Real):
            numstr = [nums].__repr__()
        else:
            raise TypeError("Not numbers.")
    except Exception as err:
        if sup:
            numstr = None
        else:
            numstr = err
    
    return numstr


def num_cat_labels(data):

    is_num = data.apply(is_numeric_dtype, axis=0).values
    is_bool = data.apply(is_bool_dtype, axis=0).values
    
    nums = to_column_choices(data.columns[is_num & (~is_bool)])
    cats = to_column_choices(data.columns[(~is_num) | is_bool])
    nbs = to_column_choices(data.columns[is_num])

    return nums, cats, nbs


def discrete_labels(data, max_cats=50):

    s = data.apply(lambda x: len(x.unique())) <= max_cats
    
    return to_column_choices(s[s].index)


def operation_source(op, name, data, ui_input, memory):

    name_out = ui_input.op_name_out_text().strip()
    left = f"{name_out} = " if name_out != '' else ""
    result = f"\n{name_out}" if name_out != "" else ""
    right = "" if left == "" else ".copy()"

    markdown = ui_input.op_markdown_text_area().strip()

    imports = []

    code = f"{left}{name}{right}{result}"
    if op == "Select columns":
        columns = to_selected_columns(ui_input.select_columns_selectize(), data)
        if columns != data.columns.tolist():
            code = (
                f"columns = {columns}\n"
                f"{left}{name}[columns].copy()"
                f"{result}"
            )

    elif op == "Filter rows":
        current_column = ui_input.filter_column_selectize()
        current_operator = ui_input.filter_operator_selectize()
        current_value_str = ui_input.filter_value_text().strip()

        filters = memory.copy()
        if current_column != "" and current_operator != "" and current_value_str != "":
            current_filter = dict(column=current_column, operator=current_operator, value=current_value_str)
            filters += [current_filter]
        
        cond_lines = []
        for i, filter in enumerate(filters):
            column = to_selected_columns(filter["column"], data)
            operator = filter["operator"]
            value = filter["value"]

            if operator in ["in", "not in"]:
                iter_values = str_to_values(value, sup=True)
                if iter_values is None:
                    iter_values = value.__repr__()
                elif isinstance(iter_values, Iterable) and not isinstance(iter_values, str):
                    iter_values = list(iter_values)
                else:
                    iter_values = [iter_values]
                not_code = "~" if operator == "not in" else ""
                opr_code = f".isin({iter_values})"
            else:
                not_code = ""
                opr_code = f" {operator} {value}"

            cond_code = f"cond{i+1} = {not_code}{name}[{column.__repr__()}]{opr_code}"
            cond_lines.append(cond_code)
        
        if len(cond_lines) > 0:
            rows_code = " & ".join([f"cond{i}"for i in range(1, 1 + len(cond_lines))])
            reset_code = ".reset_index(drop=True)" if ui_input.filter_reset_switch() else ""
            code = (
                f"{'\n'.join(cond_lines)}\n"
                f"{left}{name}.loc[{rows_code}]{reset_code}{result}"
            )

    elif op == "Sort rows":
        columns = to_selected_columns(ui_input.sort_columns_selectize(), data)
        descending = ui_input.sort_descending_switch()
        reset = ui_input.sort_reset_switch()
        if len(columns) > 0:
            descending_code = f", ascending=False" if descending else ""
            columns_code = columns[::-1].__repr__()
            reset_code = ".reset_index(drop=True)" if reset else ""
            code = f"{left}{name}.sort_values({columns_code}{descending_code}){reset_code}{result}"
    
    elif op == "Correlation":
        metric = ui_input.corr_metric_selectize()
        columns = to_selected_columns(ui_input.corr_columns_selectize(), data)
        drops = to_selected_columns(ui_input.corr_drops_selectize(), data)
        
        if columns is not None and metric != "":
            method = "corr" if metric == "Correlation" else "cov"
            drop_code = f".drop(index={drops.__repr__()})" if len(drops) > 0 else ""
            code = (
                f"columns = {columns.__repr__()}\n"
                f"{left}{name}[columns].{method}(){drop_code}{result}"
            )
    
    elif op == "Aggregation":
        columns = to_selected_columns(ui_input.agg_columns_selectize(), data)
        aggs = list(ui_input.agg_methods_selectize())

        transpose_code = ".transpose()" if ui_input.agg_transpose_switch() else ""

        if columns == data.columns.tolist():
            code = f"{left}{name}.agg({aggs}){transpose_code}{result}"
        elif len(columns) > 0:
            code = (
                f"columns = {columns}\n"
                f"{left}{name}[columns].agg({aggs}){transpose_code}{result}"
            )
    
    elif op == "Group by":
        columns = to_selected_columns(ui_input.group_by_columns_selectize(), data)
        views = to_selected_columns(ui_input.group_view_columns_selectize(), data)
        aggs = list(ui_input.group_methods_selectize())
        reset_index = ui_input.group_reset_switch()
        reset_code = ".reset_index()" if reset_index else ""
        if columns != [] and views != [] and aggs != []:
            code = (
                f"{left}{name}.groupby({columns.__repr__()})[{views.__repr__()}]"
                f".agg({aggs.__repr__()}){reset_code}{result}"
            )
    
    elif op == "Pivot table":
        values = to_selected_columns(ui_input.pivot_values_selectize(), data)
        index = to_selected_columns(ui_input.pivot_index_selectize(), data)
        columns = to_selected_columns(ui_input.pivot_columns_selectize(), data)
        methods = list(ui_input.pivot_methods_selectize())
        reset = ui_input.pivot_reset_switch()

        if values != [] and index != [] and columns != [] and methods != []:
            values_code = f"values = {values.__repr__()}"
            index_code = f"index = {index.__repr__()}"
            columns_code = f"columns = {columns.__repr__()}"
            methods_code = f"func = {methods.__repr__()}"
            reset_code = ".reset_index()" if reset else ""
            code = (
                f"{values_code}\n"
                f"{index_code}\n"
                f"{columns_code}\n"
                f"{methods_code}\n"
                f"{left}{name}.pivot_table(values, index, columns, aggfunc=func){reset_code}{result}"
            )

    elif op == "Treat missing values":
        columns = to_selected_columns(ui_input.nan_columns_selectize(), data)
        method = ui_input.nan_method_selectize()
        treat_all = len(columns) == 0
        if method == "drop":
            reset_code = ".reset_index(drop=True)" if ui_input.nan_reset_switch() else ""
            if treat_all:
                code = f"{left}{name}.dropna(){reset_code}{result}"
            else:
                code = (
                    f"missing = {name}[{columns.__repr__()}].isnull().any(axis=1)\n"
                    f"{left}{name}.loc[~missing]{reset_code}{result}"
                )
        elif method == "fill":
            value_str = ui_input.nan_fill_value_text().strip()
            if value_str == "":
                right = "" if left == "" else ".copy()"
                code = f"{left}{name}{right}{result}"
            else:
                if treat_all:
                    code = f"{left}{name}.fillna({value_str}){result}"
                else:
                    if left == "":
                        copy_code = f"{name}_copy = {name}.copy()"
                        copy_name = f"{name}_copy"
                    else:
                        copy_code = f"{left}{name}.copy()"
                        copy_name = name_out
                    code = (
                        f"{copy_code}\n"
                        f"columns = {columns.__repr__()}\n"
                        f"{copy_name}.loc[:, columns] = {name}.loc[:, columns].fillna({value_str})\n"
                        f"{copy_name}"
                    )
    
    elif op == "Time trend":
        columns = to_selected_columns(ui_input.time_trend_columns_selectize(), data)
        transform = ui_input.time_trend_transform_selectize()
        step = ui_input.time_trend_step_numeric()

        copy_name = f"{name}_copy" if left == "" else name_out
        copy_code = f"{copy_name} = {name}.copy()" if left == "" else f"{left}{name}.copy()"
        
        if len(columns) > 0 and transform != "":
            if transform == "change":
                expr = f"{copy_name}[from_cols].diff({step})"
            elif transform == "relative change":
                expr = f"{copy_name}[from_cols].pct_change({step})"
            elif transform == "log change":
                expr = f"np.log({copy_name}[from_cols]).diff({step})"
            elif transform == "moving average":
                expr = f"{copy_name}[from_cols].rolling({step}).mean()"
            else:
                expr = "None"
            step_str = f"{step}-step " if step != 1 or transform == "moving average" else ""
            code = (
                f"from_cols = {columns.__repr__()}\n"
                f"to_cols = [f'{step_str}{transform} of {{c}}' for c in from_cols]\n"
                f"{copy_code}\n"
                f"{copy_name}[to_cols] = {expr}\n"
                f"{copy_name}"
            )
    
    elif op == "Clustering":
        method = ui_input.clustering_method_selectize()
        columns = to_selected_columns(ui_input.clustering_columns_selectize(), data)
        num_str = str_to_numstr(ui_input.clustering_numbers_text(), sup=False)

        if isinstance(num_str, str) and len(columns) > 0 and method != "":
            cluster_numbers = list(eval(num_str))
            method_map = {"K-means clustering": "KMeans",
                          "Hierarchical clustering": "AgglomerativeClustering"}
            
            columns_code = f"columns = {columns.__repr__()}\n"
            copy_name = f"{name}_copy" if left == "" else f"{name_out}"
            copy_code = f"{copy_name} = {name}.copy()" if left == "" else f"{left}{name}.copy()"
            func_code = f"cluster.{method_map[method]}"

            cluster_params_code = ", max_iter=10000, random_state=0" if method == "K-means clustering" else ""
            if len(cluster_numbers) == 0:
                return f"{left}{name}.copy()"
            elif len(cluster_numbers) == 1:
                cn = cluster_numbers[0]
                fit_code = (
                    f"estimator = {func_code}(n_clusters={cn}{cluster_params_code})\n"
                    f"labels = estimator.fit_predict(scaled_features)\n"
                    f"{copy_name}['cluster_num_{cn}'] = pd.Series(labels).apply(lambda x: f'c{{x}}')"
                )
            else:
                fit_code = (
                    f"for cn in {cluster_numbers.__repr__()}:\n"
                    f"    estimator = {func_code}(n_clusters=cn{cluster_params_code})\n"
                    f"    labels = estimator.fit_predict(scaled_features)\n"
                    f"    {copy_name}[f'cluster_num_{{cn}}'] = pd.Series(labels).apply(lambda x: f'c{{x}}')"
                )

            code = (
                f"{copy_code}\n"
                f"{columns_code}"
                f"scaled_features = StandardScaler().fit_transform({name}[columns])\n"
                f"{fit_code}\n"
                f"{copy_name}"
            )
            imports.extend(["from sklearn.preprocessing import StandardScaler",
                            "from sklearn import cluster",
                            "import pandas as pd"])
    
    elif op == "Add columns":
        exp_type = ui_input.add_cols_type_selectize()
        from_columns = to_selected_columns(ui_input.add_cols_from_columns_selectize(), data)
        if isinstance(from_columns, str):
            from_columns = from_columns.__repr__()
        to_columns = ui_input.add_cols_to_columns_text().strip()
        to_columns_repr = to_columns.__repr__()

        copy_name = f"{name}_copy" if left == "" else f"{name_out}"
        copy_code = f"{copy_name} = {name}.copy()" if left == "" else f"{left}{name}.copy()"
        
        formula = ui_input.add_cols_expression_text().strip() if exp_type != "To dummies" else "None"
        transform_code = ""
        if from_columns != "" and to_columns != "" and formula != "":
            prep_code = ""
            if exp_type == "Arithmetic expression":
                expr = formula  
            elif exp_type == "Type conversion":
                expr = f"{copy_name}[{from_columns}].astype({formula})"
            elif exp_type == "String operations":
                expr = f"{copy_name}[{from_columns}].str{formula}"
            elif exp_type == "To date time":
                format_code = "" if formula == "None" or formula == "" else f", format={formula.__repr__()}"
                expr = f"pd.to_datetime({copy_name}[{from_columns}]{format_code})"
                imports.append("import pandas as pd")
            elif exp_type == "To dummies":
                drop_code = ", drop_first=True" if ui_input.add_cols_drop_switch() else ""
                cat_code = '{cat}'
                binary_code = ".astype(int)" if ui_input.add_cols_binary_switch() else ""
                prep_code = (
                    f"dummies = pd.get_dummies({copy_name}[{from_columns}]{drop_code}){binary_code}\n"
                    f"columns = [f'{to_columns}_{cat_code}' for cat in dummies.columns]\n"
                )
                expr = "dummies"
                to_columns_repr = "columns"
                imports.append("import pandas as pd")
            elif exp_type == "To segments":
                bins = str_to_values(formula)
                if isinstance(bins, Iterable):
                    bins = list(bins)
                    if len(bins) == 1:
                        bins = bins[0]
                labels = ui_input.add_cols_labels_text().strip()
                if labels != "":
                    label_list = [lab.strip() for lab in labels.split(',')]
                    labels_code = f", labels={label_list.__repr__()}"
                else:
                    labels_code = ""
                expr = f"pd.cut({copy_name}[{from_columns}], bins={bins.__repr__()}{labels_code}).astype(str)"
                imports.append("import pandas as pd")
            
            transform_code = f"\n{prep_code}{copy_name}[{to_columns_repr}] = {expr}"

        code = (
            f"{copy_code}"
            f"{transform_code}\n"
            f"{copy_name}"
        )

    source = dict(name_out=name_out, code=code, imports=imports, markdown=markdown)

    return source


def visual_source(dv, name, data, ui_input, color, memory):

    col_nums, col_cats, col_nbs = num_cat_labels(data)

    imports = ["import matplotlib.pyplot as plt"]
    markdown = ui_input.dv_markdown_text_area().strip()

    width, height = ui_input.fig_width_slider()/100, ui_input.fig_height_slider()/100
    color_code = "" if color == "#1f77b4" else f", color={color.__repr__()}"

    if dv == "Pair plot":
        fig_code = title_code = xlabel_code = ylabel_code = font_code = rotate_code = grid_code = ""
        legend_loc = ""
    else:
        fig_code = f"fig = plt.figure(figsize=({width}, {height}))\n"
        
        fontsize = ui_input.fig_fontsize_selectize()
        font_code = "" if int(fontsize) == 10 else f", fontsize={fontsize}"

        title = ui_input.fig_title_text().strip()
        title_code = f"plt.title({title.__repr__()}{font_code})\n" if title != "" else ""

        xlabel, ylabel = ui_input.fig_xlabel_text().strip(), ui_input.fig_ylabel_text().strip()
        xlabel_code = f"plt.xlabel({xlabel.__repr__()}{font_code})\n" if xlabel != "" else ""
        ylabel_code = f"plt.ylabel({ylabel.__repr__()}{font_code})\n" if ylabel != "" else ""

        legend_loc = ui_input.fig_legend_loc_selectize()
        
        rotate = ui_input.fig_xtick_rotate_numeric()
        align = "center" if rotate%90== 0 else "left" if rotate < 0 else "right"
        rotate_cond = rotate != 0
        if dv == "Bar chart":
            rotate_cond = (ui_input.bar_direction_selectize() != 'Horizontal')
        rotate_code = f"plt.xticks(rotation={rotate}, ha={align.__repr__()})\n" if rotate_cond else ""
    
        grid_code = "plt.grid()\n" if ui_input.fig_grid_switch() else ""

    plot_code = ""
    if dv == "Value counts":
        column = to_selected_columns(ui_input.value_counts_column_selectize(), data)
        direction = ui_input.value_counts_direction_selectize()
        method = ui_input.value_counts_method_selectize()
        alpha = ui_input.value_counts_alpha_slider()

        if column != "" and direction != "" and method != "":
            method_code = "" if method == "Count" else "normalize=True"
            if direction == "Vertical":
                bar_code = "bar" 
                sort_code = ""
            else:
                bar_code = "barh"
                sort_code = ".sort_values(ascending=True)"
            alpha_code = "" if alpha == 1 else f", alpha={alpha}"

            plot_code = (
                f"summary = {name}[{column.__repr__()}].value_counts({method_code}){sort_code}\n"
                f"plt.{bar_code}(summary.index, summary.values{color_code}{alpha_code})\n"
            )
    elif dv == "Histogram":
        column = to_selected_columns(ui_input.hist_column_selectize(), data)
        method = ui_input.hist_method_selectize().lower()

        if column != "" and method != "":
            bins = ui_input.hist_bins_numeric()
            hue = to_selected_columns(ui_input.hist_group_by_selectize(), data)
            norm = ui_input.hist_grouped_norm_selectize()
            style = ui_input.hist_grouped_multiple_selectize().lower()
            cmap = ui_input.hist_grouped_cmap_selectize()
            
            hue_code = f", hue={hue.__repr__()}"
            common_code = f", common_norm={norm == "Jointly"}"
            style_code = f", multiple={style.__repr__()}"
            cmap_code = f", palette={cmap.__repr__()}"
            legend_code = f"sns.move_legend(fig.gca(), loc={legend_loc.__repr__()}{font_code})\n"
            break_code = f"\n             "

            if hue == "":
                hue_code = common_code = style_code = cmap_code = legend_code = break_code = ""
            else:
                color_code = ""
            alpha_code = f", alpha={ui_input.hist_alpha_slider()}"

            plot_code = (
                f"sns.histplot({name}, x={column.__repr__()}{hue_code}, bins={bins}, {break_code}"
                f"stat={method.__repr__()}{common_code}{style_code}"
                f"{cmap_code}{color_code}{alpha_code})\n"
                f"{legend_code}"
            )
    elif dv == "KDE":
        column = to_selected_columns(ui_input.kde_column_selectize(), data)

        if column != "":
            hue = to_selected_columns(ui_input.kde_group_by_selectize(), data)
            norm = ui_input.kde_grouped_norm_selectize()
            style = ui_input.kde_grouped_multiple_selectize().lower()
            cmap = ui_input.kde_grouped_cmap()
            
            hue_code = f", hue={hue.__repr__()}"
            common_code = f", common_norm={norm == "Jointly"}"
            style_code = f", multiple={style.__repr__()}"
            cmap_code = f", palette={cmap.__repr__()}"
            legend_code = f"sns.move_legend(fig.gca(), loc={legend_loc.__repr__()}{font_code})\n"
            break_code = f"\n            "

            if hue == "":
                hue_code = common_code = style_code = cmap_code = legend_code = break_code = ""
            else:
                color_code = ""
            alpha_code = f", alpha={ui_input.kde_alpha_slider()}"

            plot_code = (
                f"sns.kdeplot({name}, x={column.__repr__()}{hue_code}, {break_code}"
                f"fill=True{common_code}{style_code}{cmap_code}{color_code}{alpha_code})\n"
                f"{legend_code}"
            )
    elif dv == "Box plot":
        column = to_selected_columns(ui_input.boxplot_column_selectize(), data)
        group = to_selected_columns(ui_input.boxplot_group_by_selectize(), data)
        hue = to_selected_columns(ui_input.boxplot_hue_selectize(), data)
        direction = ui_input.boxplot_direction_selectize()
        cmap = ui_input.boxplot_grouped_cmap_selectize()

        h, v = "x", "y"
        if direction == "Horizontal":
            h, v = v, h

        box_width = ui_input.boxplot_width_numeric()
        if column != "":
            hdata_code = f", {h}={group.__repr__()}" if group != "" else ""
            data_code = f"data={name}{hdata_code}, {v}={column.__repr__()}"
            orient_code = ", orient='h'" if direction == "Horizontal" else ""
            
            hue_code = f", hue={hue.__repr__()}"
            cmap_code = f", palette={cmap.__repr__()}"
            break_code = f"\n            "
            legend_code = f"sns.move_legend(fig.gca(), loc={legend_loc.__repr__()}{font_code})\n"
            if hue == "":
                hue_code = cmap_code = break_code = legend_code = ""
            else:
                color_code = ""

            alpha_code = f", boxprops=dict(alpha={ui_input.boxplot_alpha_slider()})"
            
            plot_code = (
                f"sns.boxplot({data_code}{orient_code}{hue_code}, {break_code}"
                f"width={box_width}{color_code}{cmap_code}{alpha_code})\n"
                f"{legend_code}"
            )

    elif dv == "Pair plot":
        xcols = to_selected_columns(ui_input.pair_columns_selectize(), data)
        drops = to_selected_columns(ui_input.pair_drop_rows_selectize(), data)
        ycols = [c for c in xcols if c not in drops]

        if len(xcols) > 0 and len(ycols) > 0:
            if len(drops) > 0:
                cols_code = (
                    f"xcols = {xcols.__repr__()}\n"
                    f"ycols = {ycols.__repr__()}\n"
                )
                vars_code = ", x_vars=xcols, y_vars=ycols"
            else:
                cols_code = f"cols = {xcols.__repr__()}\n"
                vars_code = ", vars=cols"

            hues = to_selected_columns(ui_input.pair_hue_selectize(), data)
            cmap = ui_input.pair_cmap_selectize()
            alpha = ui_input.pair_alpha_slider()
            kind = ui_input.pair_kind_selectize()
            diag = ui_input.pair_diag_kind_selectize()

            hue_code = "" if hues == "" else f", hue={hues.__repr__()}"
            cmap_code = "" if hues == "" else f", palette={cmap.__repr__()}"
            corner_code = ", corner=True" if ui_input.pair_corner_switch() else ""

            if kind == "reg":
                kws_code = f"plot_kws={{'scatter_kws': {{'alpha': {0.5*alpha}}} }}"
            else:
                kws_code = f"plot_kws={{'alpha': {alpha}, 'edgecolor': 'none'}}"

            each_width, each_height = width/len(xcols), height/len(ycols)
            all_grid_code = "[ax.grid() for ax in fig.axes]\n" if ui_input.fig_grid_switch() else ""

            plot_code = (
                f"{cols_code}"
                f"plots = sns.pairplot({name}{vars_code}{hue_code},\n"
                f"                     kind={kind.__repr__()}, diag_kind={diag.__repr__()},\n"
                f"                     {kws_code}{cmap_code},\n"
                f"                     height={each_height:.4f}, aspect={each_width/each_height:.4f}{corner_code})\n"
                "fig = plots.figure\n"
                f"{all_grid_code}"
            )
        else:
            plot_code = "fig = plt.figure()\n"

    elif dv == "Bar chart":
        current_ydata = ui_input.bar_ydata_selectize()
        current_color = color

        bars = memory.copy()
        ydata = []
        bar_colors = []
        for bar in bars:
            ydata.append(to_selected_columns(bar["ydata"], data))
            bar_colors.append(bar["color"])
        if current_ydata != "":
            ydata.append(to_selected_columns(current_ydata, data))
            bar_colors.append(current_color)
        
        if len(ydata) > 0:
            xdata = to_selected_columns(ui_input.bar_xdata_selectize(), data)
            xdata_code = "" if xdata == "" else f"x={xdata.__repr__()}, "

            bar_width = ui_input.bar_width_slider()
            bar_func = "barh" if ui_input.bar_direction_selectize() == 'Horizontal' else "bar"
            stacked_code = ", stacked=True" if ui_input.bar_mode_selectize() == "Stacked" else ""

            legend_code = "" if len(ydata) <= 1 else f"plt.legend(loc={legend_loc.__repr__()}{font_code})\n"
            hide_legend_code = "" if len(ydata) > 1 else ", legend=False"

            hide_xlabel_code = ", xlabel=''" if ui_input.fig_xlabel_text() == "" else ""
            hide_ylabel_code = ", ylabel=''" if ui_input.fig_ylabel_text() == "" else ""
            
            alpha = ui_input.bar_alpha_slider()
            alpha_code = "" if alpha == 1 else f", alpha={alpha}"
            shift = " " *(len(name) + (bar_func == "barh"))
            plot_code = (
                f"{name}.plot.{bar_func}({xdata_code}y={ydata.__repr__()}, "
                f"color={bar_colors.__repr__()}{alpha_code},\n"
                f"{shift}          width={bar_width}{stacked_code}"
                f"{hide_xlabel_code}{hide_ylabel_code}{hide_legend_code}, ax=fig.gca())\n"
                f"{legend_code}"
            )
    
    elif dv == "Line plot":
        markers = {"none": "",
                   "circle": "o", "square": "s", "diamond": "d", "triangle": "^",
                   "dot": ".", "star": "*", "cross": "x"}
        styles = {"solid": "-", "dash": "--", "dot": ":", "dash-dot": "-."}

        lines = memory.copy()
        ydata = ui_input.line_ydata_selectize()
        if ydata != "":
            lines.append(dict(xdata=ui_input.line_xdata_selectize(),
                              ydata=ydata,
                              color=color,
                              style=ui_input.line_style_selectize(),
                              marker=ui_input.line_marker_selectize(),
                              width=ui_input.line_width_slider(),
                              scale=ui_input.line_marker_scale_slider()))

        line_code = []
        for line in lines:
            ydata = to_selected_columns(line['ydata'], data)
            ydata_code = f"{name}[{ydata.__repr__()}]"
            xdata = to_selected_columns(line['xdata'], data)
            xdata_code = "" if line["xdata"] == "" else f"{name}[{xdata.__repr__()}], "
            
            color_code = f", color={line['color'].__repr__()}"
            width_code = f"linewidth={line['width']}"
            style = styles[line["style"]]
            style_code = "" if style == '-' else f", linestyle={style.__repr__()}"
            marker = markers[line["marker"]]
            marker_code = "" if marker == '' else f", marker={marker.__repr__()}"
            scale = 3**(line["scale"] - 1)
            scale_code = "" if scale == 1 else f", markersize={6*scale:.3f}"
            label_str = f"{line['ydata']}"
            each_code = (
                f"plt.plot({xdata_code}{ydata_code}{color_code},\n"
                f"         {width_code}{style_code}{marker_code}{scale_code}, label={label_str.__repr__()})"
            )
            line_code.append(each_code)

        legend_code = "" if len(line_code) < 2 else f"plt.legend(loc={legend_loc.__repr__()}{font_code})\n"
        plot_code = (
            f"{'\n'.join(line_code)}\n"
            f"{legend_code}"
        )

    elif dv == "Scatter plot":
        xdata = to_selected_columns(ui_input.scatter_xdata_selectize(), data)
        ydata = to_selected_columns(ui_input.scatter_ydata_selectize(), data)
        color_data = ui_input.scatter_color_data_selectize()

        if xdata != "" and ydata != "":
            each_code = "_each" if color_data != "" and color_data in col_cats else ""

            xdata_code = f"{name}{each_code}[{xdata.__repr__()}]"
            ydata_code = f"{name}{each_code}[{ydata.__repr__()}]"
            
            size_data = to_selected_columns(ui_input.scatter_size_data_selectize(), data)
            scale = ui_input.scatter_size_scale_slider()
            multiplier = 25**(scale-1)
            if size_data == "":
                size_code = "" if scale == 1 else f", s={36*multiplier:.3f}"
            else:
                multi_code = "" if multiplier == 1 else f"*{multiplier:.3f}"
                size_code = f", s={name}{each_code}[{size_data.__repr__()}]{multi_code}"
                
            alpha = ui_input.scatter_alpha_slider()
            alpha_code = f", alpha={alpha}"
            cmap = ui_input.scatter_cmap_selectize()
            if color_data == "":
                plot_code = (
                    f"plt.scatter({xdata_code}, {ydata_code}{size_code}{color_code}{alpha_code})\n"
                )
            elif color_data in col_nums:
                color_code = f", c={name}[{to_selected_columns(color_data, data).__repr__()}]"
                plot_code = (
                    f"plt.scatter({xdata_code}, {ydata_code}{size_code}{color_code}{alpha_code})\n"
                    f"plt.set_cmap({cmap.__repr__()})\n"
                    f"plt.colorbar()\n"
                )
            elif color_data in col_cats:
                color_col_code = f"{name}[{to_selected_columns(color_data, data).__repr__()}]"
                label_code = ", label=cat"
                plot_code = (
                    f"colors = plt.cm.{cmap}.colors\n"
                    "nc = len(colors)\n"
                    f"for i, cat in enumerate({color_col_code}.unique()):\n"
                    f"    {name}_each = {name}.loc[{color_col_code} == cat]\n"
                    f"    plt.scatter({xdata_code}, {ydata_code}{size_code}, color=colors[i%nc]"
                    f"{alpha_code}{label_code})\n"
                    f"plt.legend(loc={legend_loc.__repr__()}{font_code})\n"
                )

    elif dv == "Filled areas":
        cmap = ui_input.filled_areas_cmap_selectize()
        style = ui_input.filled_areas_style_selectize()
        ydata = to_selected_columns(ui_input.filled_areas_ydata_selectize(), data)
        xdata = to_selected_columns(ui_input.filled_areas_xdata_selectize(), data)
        alpha = ui_input.filled_areas_alpha_slider()
        
        if style != "Stack":
            bottom_init_code = ""
            y1_code = f", y1={name}[c]"
            y2_code = ""
            bottom_code = ""
        else:
            bottom_init_code = "bottom = 0\n"
            y1_code = f", y1=bottom+{name}[c]"
            y2_code = f", y2=bottom"
            bottom_code = f"    bottom += {name}[c]\n"
        
        xdata_code = f"{name}.index" if xdata == "" else f"{name}[{xdata.__repr__()}]"
        color_code = ", color=colors[i%nc]"
        alpha_code = alpha_code = f", alpha={alpha}" if alpha != 1 else ""
        label_code = f", label=c"
        legend_code = "" if len(ydata) <= 1 else f"plt.legend(loc={legend_loc.__repr__()}{font_code})\n"

        plot_code = (
            f"colors = plt.cm.{cmap}.colors\n"
            "nc = len(colors)\n"
            f"{bottom_init_code}"
            f"columns = {ydata}\n"
            "for i, c in enumerate(columns):\n"
            f"    plt.fill_between({xdata_code}{y1_code}{y2_code}{color_code}{alpha_code}{label_code})\n"
            f"{bottom_code}"
            f"{legend_code}"
        )
        
    if plot_code == "":
        config_code = ""
    else:
        config_code = (
            f"{title_code}"
            f"{xlabel_code}"
            f"{ylabel_code}"
            f"{rotate_code}"
            f"{grid_code}"
        )

    code = (
        f"{fig_code}"
        f"{plot_code}"
        f"{config_code}"
        "plt.show()"
    )

    if dv in ["Histogram", "KDE", "Box plot", "Pair plot"]:
        imports.append("import seaborn as sns")
    
    return dict(code=code, imports=imports, markdown=markdown)


def operation_exec_source(data, name, source):
    try:
        error = source["error"]
        if error is not None:
            raise RuntimeError(error)
        
        imports = source["imports"]
        code = source['code']
        name_out = source["name_out"]

        # Build exec namespace
        ns = {}
        ns[name] = data
        ns['data'] = data

        # Run imports in ns
        if imports:
            exec("\n".join(imports), ns)
        
        # Operation code execution
        if name_out == "":
            lines = code.split("\n")
            exec("\n".join(lines[:-1]), ns)
            return eval(lines[-1], ns)
        else:
            exec(code, ns)
            return eval(name_out, ns)
    except Exception as err:
        return str(err)


def visual_exec_source(data, name, dvs_dict):

    plt.close('all')
    try:
        imports = dvs_dict["source"]["imports"]
        code = dvs_dict["source"]["code"]

        ns = {}
        ns[name] = data
        ns['data'] = data

        if imports:
            exec("\n".join(imports), ns)        
        code_lines = code.split("\n")
        exec("\n".join(code_lines[:-1]), ns)
        return eval("fig", ns)
    except Exception as err:
        return str(err)

def statsmodels_source(mds_dict, name, ui_input):

    markdown = ui_input.md_markdown_text_area().strip()
    imports = ["import statsmodels.formula.api as smf"]

    func = ui_input.statsmodels_type_selectize()
    mds_dict["type"] = func
    formula = ui_input.statsmodels_formula_text().strip()

    if formula != "" and func != "":
        code = (
            f"model = smf.{func}({formula.__repr__()}, data={name})\n"
            f"result = model.fit()\n"
            f"print(result.summary())"
        )
    else:
        code = ""

    return dict(code=code, imports=imports, markdown=markdown)


def sklearn_model_source(mds_dict, name, data, ui_input, page):

    predicted = ui_input.model_dependent_selectize()
    predictors = list(ui_input.model_independent_selectize())

    code_step1 = code_step2 = code_step3 = code_step4 = ""
    imports_step1 = []

    cat_predictors = []
    if predicted != "" and len(predictors) > 0:
        _, cat_predictors, _ = num_cat_labels(data[to_selected_columns(predictors, data)])
        cat_predictors += list(ui_input.model_numeric_cats_selectize())

        if len(cat_predictors) > 0:
            imports_step1.extend(["from sklearn.preprocessing import OneHotEncoder",
                                  "from sklearn.compose import ColumnTransformer"])
            dummy_code = (
                f"\n\ncats = {cat_predictors.__repr__()}\n"
                "ohe = OneHotEncoder(drop='first', sparse_output=False)\n"
                "to_dummies = ColumnTransformer(transformers=[('cats', ohe, cats)],\n"
                "                               remainder='passthrough',\n"
                "                               force_int_remainder_cols=False)"
            )
        else:
            dummy_code = ""
        
        y = data[predicted]
        if (not is_numeric_dtype(y)) or is_bool_dtype(y):
            mds_dict["type"] = "Classifier"
        else:
            mds_dict["type"] = "Regressor"

        code_step1 = (
            f"y = {name}[{predicted.__repr__()}]\n"
            f"x = {name}[{predictors.__repr__()}]"
            f"{dummy_code}"
        )

    model = ui_input.sklearn_model_selectize()
    imports_step2 = []
    params = []
    if model != "":
        imports_step2.extend(["import numpy as np", "from sklearn.pipeline import Pipeline"])
        scaler = ui_input.sklearn_scaling_selectize()
        if scaler in ["StandardScaler", "Normalizer"]:
            imports_step2.append(f"from sklearn.preprocessing import {scaler}")
            scaler_code = "    ('scaling', StandardScaler()),\n"
        else:
            scaler_code = ""
        pca_str = str_to_numstr(ui_input.skleanr_pca_numbers())
        pca = eval(pca_str) if isinstance(pca_str, str) else []
        if len(pca) > 0:
            imports_step2.append(f"from sklearn.decomposition import PCA")
            if len(pca) == 1:
                pca_code = f"    ('pca', PCA(n_components={pca[0]}))\n"
            else:
                params.append(f"    'pca__n_components': {pca_str}")
                pca_code = "    ('pca', PCA()),\n"
        else:
            pca_code = ""
            
    if model in ["LinearRegression", "Ridge", "Lasso", "LogisticRegression"]:
        imports_step2.append(f"from sklearn.linear_model import {model}")
    elif model in ["DecisionTreeRegressor", "DecisionTreeClassifier"]:
        imports_step2.append(f"from sklearn.tree import {model}")
    elif model in ["RandomForestRegressor", "RandomForestClassifier"]:
        imports_step2.append(f"from sklearn.ensemble import {model}")

    args = []
    if model != "":
        hyper_list = model_hypers[model]
        for hyper, label, default_value in hyper_list:
            values_str = str_to_numstr(eval(f"ui_input.sklearn_{model.lower()}_{hyper}()"))
            values = eval(values_str) if isinstance(values_str, str) else []
            if len(values) == 1:
                args.append(f"{hyper}={values[0]}")
            elif len(values) > 1:
                params.append(f"    '{model.lower()}__{hyper}': {values_str}")
    
    if model in ["Lasso", "LogisticRegression"]:
        args.append("max_iter=1000000")
    elif model in ["DecisionTreeRegressor", "DecisionTreeClassifier",
                   "RandomForestRegressor", "RandomForestClassifier"]:
        args.append("random_state=0")

    if len(cat_predictors) > 0:
        dummy_code = "    ('dummy', to_dummies),\n"
    else:
        dummy_code = ""
    if model != "":
        if len(params) > 0:
            params_code = (
                "params = {\n"
                f"{',\n'.join(params)}\n"
                "}\n"
            )
        else:
            params_code = ""
        code_step2 = (
            f"{params_code}"
            "steps = [\n"
            f"{dummy_code}"
            f"{scaler_code}"
            f"{pca_code}"
            f"    ({model.lower().__repr__()}, {model}({', '.join(args)}))\n"
            "]\n"
            "pipe = Pipeline(steps)"
        )

    imports_step3 = ["from sklearn.model_selection import KFold",
                     "from sklearn.model_selection import cross_val_predict",
                     "import pandas as pd", "import numpy as np"]
    if mds_dict["type"] == "Classifier":
        scoring_code = ", scoring='roc_auc_ovr'"
        score_name = "AUC"
    else:
        scoring_code = ""
        score_name = "R-squared"
    test_set = ui_input.sklearn_test_set_switch()
    if test_set:
        split_code = (
            "x_train, x_test, y_train, y_test = "
            f"train_test_split(x, y, test_size={ui_input.sklearn_test_ratio_numeric()}, random_state=0)\n"
        )
        y_name, x_name = "y_train", "x_train"
        test_code = (
            "\n\nmodel.fit(x_train, y_train)\n"
            f"test_score = model.score(x_test, y_test{scoring_code})\n"
            "print(f'Test score: {test_score:.4f}')"
        )
        imports_step3.append("from sklearn.model_selection import train_test_split")
    else:
        split_code = ""
        y_name, x_name = "y", "x"
        test_code = ""
    if len(params) > 0:
        cv_code = (
            f"search = GridSearchCV(pipe, params{scoring_code}, cv=cv, error_score='raise', n_jobs=-1)\n"
            f"search.fit({x_name}, {y_name})\n"
            "print('Best parameters:')\n"
            "for p in params:\n"
            "    print(f\"- {p[p.index('__')+2:]}: {search.best_params_[p]}\")\n\n"
            "model = search.best_estimator_\n"
            "best_index = search.best_index_\n"
            "score = np.array([search.cv_results_[f'split{i}_test_score'][best_index]\n"
            "                  for i in range(folds)])\n"

        )
        imports_step3.append("from sklearn.model_selection import GridSearchCV")
    else:
        cv_code = (
            f"score = cross_val_score(pipe, {x_name}, {y_name}{scoring_code}, cv=cv)\n\n"
            "model = pipe\n"
        )
        imports_step3.append("from sklearn.model_selection import cross_val_score")
    
    if mds_dict["type"] == "Classifier" and page == 3:
        pred_name = "proba"
        pred_method_code = ", method='predict_proba'"
        
        y_label = ui_input.model_dependent_selectize()
        target_class = ui_input.sklearn_class_selectize()
        if is_bool_dtype(data[y_label]) and target_class in ["True", "False"]:
            target_class = eval(target_class)

        default = target_class == ""
        threshold = 0.5 if default else ui_input.sklearn_class_threshold_slider()

        if default:
            decision_cv_code = f"\n\nyhat_cv = cross_val_predict(model, {x_name}, {y_name}, cv=cv)"
            decision_test_code = "\nyhat_test = model.predict(x_test)"
        else:
            decision_cv_code = (
                f"\n\nthreshold, target = {threshold}, {target_class.__repr__()}\n"
                f"index = np.unique({y_name}).tolist().index(target)\n"
                f"y_target = y == target\n"
                f"yhat_cv = proba_cv[:, index] > threshold"
            )
            decision_test_code = f"\nyhat_test = proba_test[:, index] > threshold"
    else:
        pred_name = "yhat"
        pred_method_code = ""
        decision_cv_code = ""
        decision_test_code = ""

    
    if test_set:
        predict_func = "predict_proba" if mds_dict["type"] == "Classifier" else "predict"
        test_pred_code = (
            f"\nmodel.fit({x_name}, {y_name})\n"
            f"{pred_name}_test = model.{predict_func}(x_test)"
        )
    else:
        test_pred_code = ""
        decision_test_code = ""

    code_step3 = (
        f"folds = {ui_input.sklearn_cv_folds_numeric()}\n"
        f"cv = KFold(n_splits=folds, shuffle=True, random_state=0)\n"
        f"{split_code}"
        f"{cv_code}"
        "index=[f'fold{i}' for i in range(folds)]\n"
        f"table = pd.DataFrame({{{score_name.__repr__()}: score.round(4)}}, index=index).T\n"
        "print(f'\\n{table}')\n"
        "print(f'Cross-validation score: {score.mean():.4f}')"
        f"{test_code}\n\n"
        f"{pred_name}_cv = cross_val_predict(model, {x_name}, {y_name}{pred_method_code}, cv=cv)"
        f"{test_pred_code}"
        f"{decision_cv_code}"
        f"{decision_test_code}"
    )

    markdown = ui_input.md_markdown_text_area()

    return dict(code={1: code_step1, 2: code_step2, 3: code_step3},
                imports={1: imports_step1, 2: imports_step2, 3: imports_step3},
                markdown=markdown)


def statsmodels_outputs_source(ui_input):

    imports = ["import pandas as pd"]
    name_out = ui_input.statsmodels_output_text().strip()

    code = (
        f"{name_out} = pd.concat((result.params, result.bse, result.tvalues, result.pvalues), axis=1)\n"
        f"{name_out}.columns = ['coef', 'std err', 't-values', 'p-vlaues']\n"
        f"{name_out}[['CI-lower', 'CI-upper']] = result.conf_int()\n"
        f"{name_out}"
    )

    return dict(type="data", name_out=name_out, code=code, imports=imports)


def sklearn_outputs_source(mds_dict, name, data, ui_input):

    imports = []
    name_out = ui_input.sklearn_output_text().strip()

    test_set = ui_input.sklearn_test_set_switch()
    row_index = "x_train.index" if test_set else ":"
    y_label = ui_input.model_dependent_selectize() 
    if mds_dict["type"] == "Classifier":
        predicted = "proba"
        classes = np.unique(data[y_label]).tolist()
        pred_cols = [f"{y_label}_proba_{c}" for c in classes]

        target_class = ui_input.sklearn_class_selectize()
        if is_bool_dtype(data[y_label]) and target_class in ["True", "False"]:
            target_class = eval(target_class)
        default = target_class == ""
        
        if default:
            label = f"{y_label}_pred".__repr__()
            decision_cv_code = f"\n{name_out}.loc[{row_index}, {label}] = yhat_cv"
            decision_test_code = f"{name_out}.loc[x_test.index, {label}] = yhat_test\n"
        else:
            label = f"{y_label}_is_{target_class}".__repr__()
            decision_cv_code = f"\n{name_out}.loc[{row_index}, {label}] = proba_cv[:, index] > threshold"
            decision_test_code = f"{name_out}.loc[x_test.index, {label}] = proba_test[:, index] > threshold\n"
    else:
        predicted = "yhat"
        pred_cols = f"{y_label}_pred"
        decision_cv_code = ""
        decision_test_code = ""

    if test_set:
        save_test_code = (
            f"\n{name_out}.loc[x_test.index, {pred_cols.__repr__()}] = {predicted}_test\n"
            f"{decision_test_code}"
            f"{name_out}.loc[x_train.index, 'split'] = 'cross-validation'\n"
            f"{name_out}.loc[x_test.index, 'split'] = 'test'"
        )
    else:
        save_test_code = ""
    
    code = (
        f"{name_out} = {name}.copy()\n"
        f"{name_out}.loc[{row_index}, {pred_cols.__repr__()}] = {predicted}_cv"
        f"{decision_cv_code}"
        f"{save_test_code}"
    )

    return dict(type="data", name_out=name_out, code=code, imports=imports)


def sklearn_plots_source(mds_dict, name, data, ui_input, page):

    y_label = ui_input.model_dependent_selectize() 
    test_set = ui_input.sklearn_test_set_switch()
    plots = ui_input.sklearn_outputs_checkbox()

    if mds_dict["type"] == "Classifier" and page == 3:
        target_class = ui_input.sklearn_class_selectize()
        if is_bool_dtype(data[y_label]) and target_class in ["True", "False"]:
            target_class = eval(target_class)
    
    test_min = test_max = cv_label_code = test_resid_code = test_plot_code = legend_code = ""
    if test_set:
        x_name, y_name = "x_train", "y_train"
        cv_label_code = ", label='Cross-validation'"
        legend_code = "plt.legend()\n"
        num_axes = 2
    else:
        x_name, y_name = "x", "y"
        num_axes = 1

    imports = ["import matplotlib.pyplot as plt"]
    source = []
    if "Prediction plot" in plots and page == 3:
        if test_set:
            test_min, test_max = ", yhat_test.min()", ", yhat_test.max()"
            test_plot_code = (
                "plt.scatter(yhat_test, y_test, linewidth=2,\n"
                "            edgecolor='r', facecolor='none', alpha=0.3, label='Test')\n"
            )
        code = (
            "fig = plt.figure(figsize=(4.2, 4.2))\n"
            f"ymin = min(yhat_cv.min(){test_min}, {y_name}.min())\n"
            f"ymax = max(yhat_cv.max(){test_max}, {y_name}.max())\n"
            f"plt.scatter(yhat_cv, {y_name}, linewidth=2,\n"
            f"            edgecolor='b', facecolor='none', alpha=0.3{cv_label_code})\n"
            f"{test_plot_code}"
            "plt.plot([ymin, ymax], [ymin, ymax], linestyle='--', color='k')\n"
            "plt.xlabel('Predicted values')\n"
            "plt.ylabel('Actual values')\n"
            f"{legend_code}"
            "plt.grid()\n"
            "plt.show()"
        )
        source.append(dict(type="plot", code=code, imports=imports, fig=None))
    
    if "Residual plot" in plots and page == 3:
        if test_set:
            test_resid_code = "resid_test = y_test - yhat_test\n"
            test_plot_code = (
                "plt.scatter(yhat_test, resid_test, linewidth=2,\n"
                "            edgecolor='r', facecolor='none', alpha=0.3, label='Test')\n"
            )
            ymin_code = "ymin = min((yhat_cv.min(), yhat_test.min()))\n"
            ymax_code = "ymax = max((yhat_cv.max(), yhat_test.max()))\n"
        else:
            ymin_code = "ymin = yhat_cv.min()\n"
            ymax_code = "ymax = yhat_cv.max()\n"
        code = (
            "fig = plt.figure(figsize=(4.2, 4.2))\n"
            f"resid_cv = {y_name} - yhat_cv\n"
            f"{test_resid_code}"
            f"{ymin_code}"
            f"{ymax_code}"
            f"plt.scatter(yhat_cv, resid_cv, linewidth=2,\n"
            f"            edgecolor='b', facecolor='none', alpha=0.3{cv_label_code})\n"
            f"{test_plot_code}"
            "plt.plot([ymin, ymax], [0, 0], linestyle='--', color='k')\n"
            "plt.xlabel('Predicted values')\n"
            "plt.ylabel('Residuals')\n"
            f"{legend_code}"
            "plt.grid()\n"
            "plt.show()"
        )
        source.append(dict(type="plot", code=code, imports=imports, fig=None))

    if "Confusion matrix" in plots and page == 3:
        
        imports.extend(["import seaborn as sns",
                        "from sklearn.metrics import confusion_matrix"])

        rows = "[x_train.index]" if test_set else ""
        default = target_class == ""
        classes = np.unique(data[y_label]).tolist()
        if default:
            args = f"{y_name}, yhat_cv"
            index_code = f"index={classes.__repr__()},"
            columns_code = f"columns={classes.__repr__()}"
        else:
            args = f"y_target{rows}, proba_cv[:, index]>threshold"
            index_code = f"index=['is {target_class}', 'not {target_class}'],"
            columns_code = f"columns=['is {target_class}', 'not {target_class}']"

        code = (
            f"cmat_cv = pd.DataFrame(confusion_matrix({args}, normalize='true').round(5),\n"
            f"                       {index_code}\n"
            f"                       {columns_code})\n"
            "cmat_cv.index.name = 'Actual'\n"
            "cmat_cv.columns.name = 'Predicted'\n"
            f"fig = plt.figure(figsize=(4.2, 4.5))\n"
            "sns.heatmap(cmat_cv, annot=True, cmap='YlGn', cbar=False, ax=fig.gca())\n"
            "plt.title('Cross-validation')\n"
            "plt.show()"
        )
        source.append(dict(type="plot", code=code, imports=imports, fig=None))

        if test_set:
            default = target_class == ""
            if default:
                args = f"y_test, yhat_test"
            else:
                args = "y_target[x_test.index], proba_test[:, index]>threshold"
            code = (
                f"cmat_test = pd.DataFrame(confusion_matrix({args}, normalize='true').round(5),\n"
                f"                         {index_code}\n"
                f"                         {columns_code})\n"
                "cmat_test.index.name = 'Actual'\n"
                "cmat_test.columns.name = 'Predicted'\n"
                f"fig = plt.figure(figsize=(4.2, 4.5))\n"
                "sns.heatmap(cmat_test, annot=True, cmap='YlGn', cbar=False, ax=fig.gca())\n"
                "plt.title('Test')\n"
                "plt.show()"
            )
            source.append(dict(type="plot", code=code, imports=imports, fig=None))
    
    if "Receiver-operating characteristic" in plots and page == 3:

        imports.append("from sklearn.metrics import roc_curve")

        rows = "[x_train.index]" if test_set else ""
        code = (
            f"fpr, tpr, thresholds = roc_curve(y_target{rows}, proba_cv[:, index])\n"
            "fig = plt.figure(figsize=(4.2, 4.5))\n"
            "plt.fill_between(fpr, tpr, color='orange', alpha=0.3, zorder=0, label='AUC')\n"
            "plt.plot(fpr, tpr, linewidth=2, color='b', zorder=1, label='ROC')\n"
            "k = np.argmin(abs(thresholds - threshold))\n"
            "plt.scatter(fpr[k], tpr[k], s=80, linewidth=2, edgecolor='b', facecolor='lightblue')\n"
            "plt.plot([0, 1], [0, 1], linewidth=2, linestyle='--', color='k')\n"
            "plt.title('Cross-validation')\n"
            "plt.legend()\n"
            "plt.xlabel('False positive rate')\n"
            "plt.ylabel('True positive rate')\n"
            "plt.grid()\n"
            "plt.show()"
        )
        source.append(dict(type="plot", code=code, imports=imports, fig=None))
    
    if "Precision-recall" in plots:

        imports.extend(["from sklearn.metrics import precision_recall_curve",
                        "from sklearn.metrics import f1_score"])
        
        rows = "[x_train.index]" if test_set else ""
        code = (
            f"precision, recall, thresholds = precision_recall_curve(y_target{rows}, proba_cv[:, index])\n"
            f"f1 = f1_score(y_target{rows}, proba_cv[:, index] > threshold)\n"
            "fig = plt.figure(figsize=(4.2, 4.5))\n"
            "plt.plot(recall, precision, linewidth=2, color='b', zorder=0)\n"
            "k = np.argmin(abs(thresholds - threshold))\n"
            "plt.scatter(recall[k], precision[k], s=80, linewidth=2, edgecolor='b', facecolor='lightblue')\n"
            "plt.text(0.6, 0.98, f'f1-score: {f1:.4f}',\n"
            "         bbox=dict(facecolor='wheat', edgecolor='black', boxstyle='round,pad=0.5'))\n"
            "plt.title('Cross-validation')\n"
            "plt.xlabel('Recall')\n"
            "plt.ylabel('Precision')\n"
            "plt.grid()\n"
            "plt.show()"
        )
        source.append(dict(type="plot", code=code, imports=imports, fig=None))

    return source


def export(nodes):

    code_cells = []
    all_imports = []
    for node in nodes:
        info = node.info
        source = info["source"]
        code = source["code"]
        if isinstance(code, dict):
            keys = ["vars", "dummy", "pipeline", "fitting"]
            code = '\n'.join([code[k] for k in keys])
        markdown = source["markdown"]
        imports = source["imports"]
    
        all_imports.extend(imports)
        
        if markdown.strip() != "":
            markdown_dict = dict(cell_type="markdown", metadata={}, source=f"{markdown}")
            code_cells.append(markdown_dict)

        code_dict = dict(cell_type='code', metadata={}, source=f'{code}')
        code_cells.append(code_dict)
    
    all_imports = list(set(all_imports))
    if "import pandas as pd" in all_imports:
        all_imports.remove("import pandas as pd")
    if "import numpy as np" in all_imports:
        all_imports.remove("import numpy as np")
    all_imports.sort(reverse=True)

    all_imports = ['import pandas as pd', 'import numpy as np'] + all_imports
    
    import_cell = dict(cell_type='code', metadata={}, source='\n'.join(all_imports))
    code_cells.insert(0, import_cell)

    return {
        "metadata" : {
            "signature": "hex-digest",
            "kernel_info": {
                "name" : "the name of the kernel"
            },
        },
        "nbformat": 4,
        "nbformat_minor": 0,
        "cells" : code_cells,
    }
