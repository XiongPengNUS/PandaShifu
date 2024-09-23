import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from collections.abc import Iterable
from keyword import iskeyword

def isinvalid(name):

    if name is None:
        return True
    else:
        return not name.isidentifier() or iskeyword(name)


def index_labels(index):

    if isinstance(index, pd.MultiIndex):
        labels = pd.Series({'_'.join([str(item) for item in i]): i for i in index})
    else:
        labels = pd.Series({str(i): i for i in index})

    return labels


def columns_code(data, output_name, input_name, columns):

    left = '' if output_name is None or output_name == '' else f'{output_name} = '
    column_labels = index_labels(data.columns)

    code = f'columns = {list(column_labels[columns])}\n'
    code += f'{left}{input_name}[columns]\n'
    if len(columns) == len(column_labels):
        if (np.array(columns) == column_labels.index).all():
            code = f'{left}{input_name}\n'

    return f'```python\n{code}```'


def filter_code(output_name, input_name, flist, reset_index):

    left = '' if output_name is None or output_name == '' else f'{output_name} = '

    conds = []
    for i, f in enumerate(flist):
        if f['column'] is None:
            continue
        if len(f['values']) == 1:
            conds.append(f"cond{i+1} = {input_name}[{f['column'].__repr__()}] == {f['values'][0].__repr__()}")
        elif len(f['values']) > 1:
            conds.append(f"cond{i+1} = {input_name}[{f['column'].__repr__()}].isin({f['values']})")
            # conds = cond & (filter_on.isin(f['values']))
        if f['range'] is not None:
            fmin, fmax = f['range']
            filter_on = f"{input_name}[{f['column'].__repr__()}]"
            conds.append(f"cond{i+1} = ({filter_on} >= {fmin}) & ({filter_on} <= {fmax})")

    if len(conds) == 0:
        code = f'{left}{input_name}\n'
    else:
        code = '\n'.join(conds)
        conds_string = ' & '.join([f'cond{i+1}' for i in range(len(conds))])
        reset_string = '.reset_index(drop=True)' if reset_index else ''
        code += f'\n{left}{input_name}.loc[{conds_string}]{reset_string}\n'

    return f'```python\n{code}```'


def treat_na_code(output_name, input_name, column_labels,
                  method, all_switch, columns, values):

    left = '' if output_name is None or output_name == '' else f'{output_name} = '
    if len(all_switch) > 0:
        if method == 'drop':
            code = f'{left}{input_name}.dropna()\n'
        elif method == 'fill':
            code = f'{left}{input_name}.fillna({values})\n'
        else:
            code = f'{input_name}\n'
    else:
        if columns is None:
            code = f'{input_name}\n'
        else:
            col = list(column_labels[columns]).__repr__()
            if method == 'drop':
                code = f'has_na = {input_name}[{col}].isnull().any(axis=1)\n'
                code += f'{left}{input_name}.loc[~has_na]\n'
            elif method == 'fill':
                if left != '':
                    code = f'{output_name} = {input_name}.copy()\n'
                    input_name = output_name
                else:
                    code = ''
                code += f'subset = {input_name}.loc[:, {col}].fillna({values})\n'
                code += f'{input_name}.fillna(subset, inplace=True)\n'
            else:
                code = f'{input_name}\n'
            
    return f'```python\n{code}```'


def sort_code(output_name, input_name, columns, ascendings, reset_index):

    left = '' if output_name is None or output_name == '' else f'{output_name} = '

    if len(columns) > 0:
        sort_code = f'.sort_values({columns[::-1]}, ascending={ascendings[::-1]})'
        if reset_index:
            sort_code += '.reset_index(drop=True)'
    else:
        sort_code = ''

    code = f'{left}{input_name}{sort_code}\n'

    if len(code) > 75:
        sep_idx = code.find(' ascending')
        if sep_idx != -1:
            indent = (len(left) + len(input_name) + 13) * ' '
        code = code[:sep_idx] + '\n' + indent + code[sep_idx+1:]

    return f'```python\n{code}```'


def grouped_code(output_name, input_name, columns_by, columns_view, apply, reset):

    left = '' if output_name is None or output_name == '' else f'{output_name} = '

    reset_code = '.reset_index()' if reset else ''
    code = f'{left}{input_name}.groupby({columns_by})[{columns_view}].agg({apply}){reset_code}\n'

    return f'```python\n{code}```'


def pivot_code(output_name, input_name, value_cols, index_cols, column_cols, agg, reset):

    left = '' if output_name is None or output_name == '' else f'{output_name} = '

    reset_code = '.reset_index()' if reset else ''
    code = f'{left}{input_name}.pivot_table({value_cols}, {index_cols}, {column_cols}, {agg}){reset_code}\n'

    return f'```python\n{code}```'


def add_colum_code(output_name, input_name, column_labels,
                   new_column, etype, from_column,
                   exprs, dtype, time_format, drop_first):
    
    if  not new_column or etype is None:
        return f'```python\n{input_name}\n```'
    non_cond1 =  etype == 'arithmetic operations' and not exprs
    non_cond2 =  etype != 'arithmetic operations' and from_column is None
    if non_cond1 or non_cond2:
        return f'```python\n{input_name}\n```'

    if etype in ['to datetime', 'to dummies']:
        code = 'import pandas as pd\n\n'
    else:
        code = ''

    if output_name is None or isinvalid(output_name):
        output_name = input_name
        code += ''
    else:
        code += f'{output_name} = {input_name}.copy()\n'
    
    left = f'{output_name}[{new_column.__repr__()}]'
    if from_column:
        cols = f'{input_name}[{column_labels[from_column].__repr__()}]'
    else:
        cols = ''
    if etype == 'type conversion':
        right = f'{cols}.astype({dtype})'
        code +=  f'{left} = {right}\n'
    elif etype == 'arithmetic operations':
        code += f'{left} = {exprs}\n'
    elif etype == 'string operations':
        right = f'{cols}.str{exprs}'
        code += f'{left} = {right}\n'
    elif etype == 'to datetime':
        right = f'{cols}, format={time_format.__repr__()}'
        code += f'{left} = pd.to_datetime({right})\n'
    elif etype == 'to dummies':
        drop_first_code = ', drop_first=True' if len(drop_first) > 0 else ''
        code += f'dummies = pd.get_dummies({cols}{drop_first_code})\n'

    return f'```python\n{code}```'


def univariate_code(input_name, col,
                    ptype, column, horizontal, density, width, nbins, color, opacity,
                    fig_width, fig_height, grid, tickangle):

    code = f'col = {input_name}[{col.name.__repr__()}]\n'
    if ptype == 'numerical results':
        code = 'import pandas as pd\n\n' + code
        if is_numeric_dtype(col) and not is_bool_dtype(col):
            code += f'{input_name}_res = pd.DataFrame(col.describe()).T\n'
            code += f'{input_name}_res[\'count\'] = {input_name}_res[\'count\'].astype(int)\n'
            code += f'{input_name}_res[\'missing\'] = col.isnull().sum()\n'
            code += f'{input_name}_res[\'dtype\'] = str(col.dtype)\n'
            code += f'{input_name}_res\n'
        else:
            code += 'counts = col.value_counts().sort_values()\n'
            code += 'pr = col.value_counts(normalize=True).values\n'
            code += input_name + '_res = pd.DataFrame({\'count\': [col.count()],\n'
            ns = len(input_name) + 1
            code += ' '*ns + '                    \'categories\': [len(counts)],\n'
            code += ' '*ns + '                    \'Gini\': [(pr * (1-pr)).sum()],\n'
            code += ' '*ns + '                    \'entropy\': [-(pr * np.log2(pr)).sum()],\n'
            code += ' '*ns + '                    \'smallest\': [f"{counts.index[0]}: {counts.values[0]}"],\n'
            code += ' '*ns + '                    \'largest\': [f"{counts.index[-1]}: {counts.values[-1]}"],\n'
            code += ' '*ns + '                    \'missing\': [col.isnull().sum()],\n'
            code += ' '*ns + '                    \'dtype\': [str(col.dtype)]}, index=[col.name])\n'
            code += f'{input_name}_res\n'
    else:
        normal = len(density) > 0
        config_code = f'plt.figure(figsize=({fig_width/80:.3f}, {fig_height/80:.3f}))\n'
        code = 'import matplotlib.pyplot as plt\n\n' + code
        if ptype == 'value counts':
            if len(horizontal) > 0:
                func = 'barh'
                width_kw = 'height'
                xlabel = 'Density' if len(density) else 'Count'
                ylabel = column
                sort = '.sort_values(ascending=True)'
            else:
                func = 'bar'
                width_kw = 'width'
                ylabel = 'Density' if len(density) else 'Count'
                xlabel = column
                sort = ''
            code += f'res = col.value_counts(normalize={normal}){sort}\n\n'
            code += config_code
            code += f'plt.{func}(res.index, res.values, {width_kw}={width}, color=\'{color}\', alpha={opacity})\n'
            code += f'plt.xlabel({xlabel.__repr__()})\n'
            code += f'plt.ylabel({ylabel.__repr__()})\n'
        elif ptype == 'hist':
            ylabel = 'Density' if len(density) else 'Count'
            code += config_code
            code += f'plt.hist(col, bins={nbins}, density={normal}, color=\'{color}\', alpha={opacity})\n'
            code += f'plt.xlabel({column.__repr__()})\n'
            code += f'plt.ylabel({ylabel.__repr__()})\n'
        elif ptype == 'kde':
            code += config_code
            code += f'sns.kdeplot(col, fill=True, color=\'{color}\', alpha={opacity})\n'
            code = 'import seaborn as sns\n' + code
        elif ptype == 'boxplot':
            vert, xy = (False, "x") if len(horizontal) else (True, "y")
            code += config_code
            code += f'plt.boxplot(col, labels=[\'\'], vert={vert}, patch_artist=True,\n'
            code += f'            boxprops=dict(facecolor=\'{color}\', alpha={opacity}))\n'
            code += f'plt.{xy}label({column.__repr__()})\n'
        else:
            code = 'To be developed.\n'

        code += f'plt.xticks(rotation={tickangle})\n'
        if len(grid) > 0:
            code += f'plt.grid()\n'
        code += 'plt.show()\n'

    return f'```python\n{code}```'


def barchart_code(input_name, input_data,
                  blist, xdata, bwidth, opacity, horizontal, btype,
                  legpos, xlabel, ylabel,
                  fig_width, fig_height, grid, tickangle):

    columns = []
    colors = []
    for b in blist:
        if b['column'] is None:
            continue
        bar_column = b['column']
        if isinstance(bar_column, list):
            bar_column = tuple(bar_column)
        columns.append(bar_column)
        colors.append(b['color'])
    if len(columns) == 0:
        return f'```python\n{input_name}\n```'

    column_labels = index_labels(input_data.columns)
    func = 'barh' if len(horizontal) > 0 else 'bar'
    xcode = '' if xdata is None else f'x={column_labels[xdata].__str__()}, '
    typecode = 'stacked=True, ' if btype == 'stacked' else ''

    code = 'import matplotlib.pyplot as plt\n\n'
    code += f'{input_name}.plot.{func}({xcode}y={columns},\n'
    space = ' ' * (len(input_name) + 10 + (len(horizontal) > 0))
    code += f'{space}color={colors}, alpha={opacity}, {typecode}width={bwidth}, legend=False,\n'
    code += f'{space}figsize=({fig_width/80:.3f}, {fig_height/80:.3f}))\n'
    code += f'plt.legend(loc={legpos.__repr__()})\n'
    if xlabel is None:
        xlabel = ''
    if ylabel is None:
        ylabel = ''
    code += f'plt.xlabel({xlabel.__repr__()})\n'
    code += f'plt.ylabel({ylabel.__repr__()})\n'
    code += f'plt.xticks(rotation={tickangle})\n'
    if len(grid) > 0:
        code += f'plt.grid()\n'
    code += 'plt.show()\n'

    return f'```python\n{code}```'


def scatterplot_code(input_name, input_data, xdata, ydata,
                     size_switch, scale, sdata, 
                     color_switch, color, cdata, opacity,
                     legpos, xlabel, ylabel,
                     fig_width, fig_height, grid, tickangle):
    
    if not xdata or not ydata:
        return f'```python\n{input_name}\n```'
    
    column_labels = index_labels(input_data.columns)
    xcol = column_labels[xdata]
    ycol = column_labels[ydata]
    
    code = 'import matplotlib.pyplot as plt\n\n'
    code += f'plt.figure(figsize=({fig_width/80:.3f}, {fig_height/80:.3f}))\n'
    if len(size_switch) > 0 and sdata:
        scale_code = f'*{5**scale:.2f}' if scale != 0 else ''
        if input_data[column_labels[sdata]].isnull().sum() > 0:
            na_code = '.fillna(0)'
        else:
            na_code = ''
        size_code = f's={input_name}[{column_labels[sdata].__repr__()}]{na_code}{scale_code}'
    else:
        size_code = 's=36' if scale == 0 else f's={36 * 5**scale:.2f}'
        na_code = ''
    if len(color_switch) > 0 and cdata:
        color_col = input_data[column_labels[cdata]]
        if is_numeric_dtype(color_col) and not is_bool_dtype(color_col):
            color_code = f'c={input_name}[{column_labels[cdata].__repr__()}]'
            color_bar = True
        else:
            color_code = None
            color_bar = False
    else:
        color_code = f'c={color.__repr__()}'
        color_bar = False
    
    if color_code:
        xs = f'{input_name}[{xcol.__repr__()}]'
        ys = f'{input_name}[{ycol.__repr__()}]'        
        code += f'plt.scatter({xs}, {ys}, {size_code}, {color_code}, alpha={opacity})\n'
        if color_bar:
            code += 'plt.colorbar()\n'
    else:
        code += f'cdata = {input_name}[{column_labels[cdata].__repr__()}]\n'
        code += f'for cat in cdata.unique():\n'
        code += f'    subset = {input_name}.loc[cdata == cat]\n'
        code += f'    plt.scatter(subset[{xcol.__repr__()}], subset[{ycol.__repr__()}], '
        if len(size_switch) > 0 and sdata:
            size_code = f's=subset[{column_labels[sdata].__repr__()}]{na_code}{scale_code}'
        else:
            size_code = 's=36' if scale == 0 else f's={36 * 5**scale:.2f}'
        code += f'{size_code}, alpha={opacity}, label=cat)\n'
        code += f'plt.legend(loc={legpos.__repr__()})\n'
    
    if xlabel is None:
        xlabel = ''
    if ylabel is None:
        ylabel = ''
    code += f'plt.xlabel({xlabel.__repr__()})\n'
    code += f'plt.ylabel({ylabel.__repr__()})\n'
    code += f'plt.xticks(rotation={tickangle})\n'
    if len(grid) > 0:
        code += f'plt.grid()\n'
    code += 'plt.show()\n'

    return f'```python\n{code}```'


def lineplot_code(input_name, llist,
                  legpos, xlabel, ylabel,
                  fig_width, fig_height, grid, tickangle):

    count = 0
    code = 'import matplotlib.pyplot as plt\n\n'
    code += f'plt.figure(figsize=({fig_width/80:.3f}, {fig_height/80:.3f}))\n'
    for lin in llist:
        if lin['y'] is None:
            continue
        count += 1
        if isinstance(lin['y'], Iterable) and not isinstance(lin['y'], str):
            lin['y'] = tuple(lin['y'])
            data_label = '_'.join(pd.Series(lin['y']).astype(str))
        else:
            data_label = f"{lin['y']}"
        if lin['trans'] == 'change':
            tcode = f".diff({lin['period']})"
            data_label = f"{lin['period']}-period change of {data_label}"
        elif lin['trans'] == 'fractional change':
            tcode = f".pct_change({lin['period']})"
            data_label = f"{lin['period']}-period fractional change of {data_label}"
        elif lin['trans'] == 'moving average':
            tcode = f".rolling({lin['period']}).mean()"
            data_label = f"{lin['period']}-period moving average of {data_label}"
        else:
            tcode = ''
        ycode = f"{input_name}[{lin['y'].__repr__()}]{tcode}"
        if not lin['x']:
            xcode = ''
        else:
            if isinstance(lin['x'], Iterable) and not isinstance(lin['y'], str):
                lin['x'] = tuple(lin['x'])
            xcode = f"{input_name}[{lin['x'].__repr__()}], "

        styles = {'solid': '-',
                  'dash': '--',
                  'dot': '.',
                  'dashdot': '-.'}
        lscode = f"linestyle={styles[lin['linestyle']].__repr__()}"
        lwcode = f"linewidth={lin['linewidth']}"
        
        if lin['marker'] == 'none' or lin['marker'] is None:
            mcode = ''
        else:
            symbols = {'circle': 'o',
                       'dot': '.',
                       'square': 's',
                       'diamond': 'd',
                       'triangle': '^'}
            mcode = f" marker={symbols[lin['marker']].__repr__()},"
        if lin['scale'] == 0 or mcode == '':
            scode = ''
        else:
            scode = f" markersize={6*5**(lin['scale']/2):.2f},"
        
        cocode = f"color={lin['color'].__repr__()}"
        lacode = f"label={data_label.__repr__()}"

        code += f'plt.plot({xcode}{ycode}, {cocode},{mcode}{scode}\n'
        code += f'         {lscode}, {lwcode}, {lacode})\n'

    if count == 0:
        return f'```python\n{input_name}\n```'
    elif count > 1:
        code += f'plt.legend(loc={legpos.__repr__()})\n'

    if xlabel is None:
        xlabel = ''
    if ylabel is None:
        ylabel = ''
    code += f'plt.xlabel({xlabel.__repr__()})\n'
    code += f'plt.ylabel({ylabel.__repr__()})\n'
    code += f'plt.xticks(rotation={tickangle})\n'
    if len(grid) > 0:
        code += f'plt.grid()\n'
    code += 'plt.show()\n'

    return f'```python\n{code}```'
    

def model_vars_code(name, predicted, predictors, mtype):

    if mtype == 'Predictive modeling':
        code = f'y = {name}[{predicted.__repr__()}]\n'
        code += f'x = {name}[{predictors}]\n'
    elif mtype == 'Explanatory modeling':
        code = f'{name}\n'
    else:
        code = '\n'
    
    return f'```python\n{code}```'


def exp_fit_code(formula, name, ftype):

    code = 'import statsmodels.formula.api as smf\n\n'
    if ftype == 'ols':
        code += f'model = smf.ols({formula.__repr__()}, data={name})\n'
    elif ftype == 'logit':
        code += f'model = smf.logit({formula.__repr__()}, data={name})\n'
    else:
        return f'```python\n{name}\n```'
    
    code += 'results = model.fit()\n'
    code += 'print(results.summary())\n'

    return f'```python\n{code}```'


def pred_cats_code(all_cats):

    if len(all_cats) > 0:
        code = 'from sklearn.preprocessing import OneHotEncoder\n'
        code += 'from sklearn.compose import ColumnTransformer\n\n'
        code += f'cats = {all_cats}\n'
        code += 'ohe = OneHotEncoder(drop=\'first\', sparse_output=False)\n'
        code += 'to_dummies = ColumnTransformer(transformers=[(\'cats\', ohe, cats)],\n'
        code += '                               remainder=\'passthrough\')\n'
    else:
        code = ''
    
    return f'```python\n{code}```'


def pred_reg_model_code(steps, params):

    imports = {'StandardScaler()': 'from sklearn.preprocessing import StandardScaler',
               'Normalizer()': 'from sklearn.preprocessing import Normalizer',
               'PCA()': 'from sklearn.decomposition import PCA',
               'LinearRegression()': 'from sklearn.linear_model import LinearRegression',
               'LogisticRegression(max_iter=1000000)': 'from sklearn.linear_model import LogisticRegression',
               'Ridge(max_iter=1000000)': 'from sklearn.linear_model import Ridge',
               'Lasso(max_iter=1000000)': 'from sklearn.linear_model import Lasso',
               'DecisionTreeRegressor(random_state=0)': 'from sklearn.tree import DecisionTreeRegressor',
               'RandomForestRegressor(random_state=0)': 'from sklearn.ensemble import RandomForestRegressor',
               'DecisionTreeClassifier(random_state=0)': 'from sklearn.tree import DecisionTreeClassifier',
               'RandomForestClassifier(random_state=0)': 'from sklearn.ensemble import RandomForestClassifier'}
    import_code = ['from sklearn.pipeline import Pipeline']
    step_code = []
    if 'dummy' in steps:
        step_code.append(f'    (\'dummy\', to_dummies)')
    
    keys = ['scaling', 'pca', 'model']
    for key in keys:
        if key in steps:
            func = steps[key].__str__()
            import_code.append(imports[func])
            step_code.append(f'    ({key.__repr__()}, {func})')
    
    params_string = ',\n'.join([f'    {key.__repr__()}: {value}'
                               for key, value in params.items()])
    if params_string == '':
        params_code = ''
    else:
        params_code = f'params = {{\n{params_string}\n}}\n'
    
    code = "{}\n\n".format('\n'.join(import_code))
    code += f"{params_code}"
    code += "steps = [\n{}\n]\n".format(',\n'.join(step_code))
    code += 'pipe = Pipeline(steps)\n'
    
    return f'```python\n{code}```'


def pred_fit_code(params, num_folds, test_switch, ratio, pred_type='R'):

    if params is None:
        return ''

    import_code = ['import numpy as np',
                   'from sklearn.model_selection import KFold']
    if pred_type == 'C':
        score_code = ', scoring=\'roc_auc\''
    else:
        score_code = ''
    if len(test_switch) > 0:
        import_code.append('from sklearn.model_selection import train_test_split')
        xy_code = 'x_train, x_test, y_train, y_test '
        xy_code += f'= train_test_split(x, y, test_size={ratio}, random_state=0)\n'
        test_code = ('\nmodel.fit(x_train, y_train)\n'
                     f'test_score = model.score(x_test, y_test{score_code})\n')
        print_test_code = 'print(f\'Test score: {test_score}\')\n'
    else:
        xy_code = 'x_train, y_train = x, y\n'
        test_code = ''
        print_test_code = ''
    if len(params) == 0:
        import_code.append('from sklearn.model_selection import cross_val_score')
        cv_code = f'scores = cross_val_score(pipe, x_train, y_train, cv=cv{score_code})\n'
        model_code = 'model = pipe\n'
        params_code = ''
    else:
        import_code.append('from sklearn.model_selection import GridSearchCV')
        cv_code = (f'search = GridSearchCV(pipe, params, cv=cv{score_code}, n_jobs=-1)\n'
                   'search.fit(x_train, y_train)\n'
                   'best_index = search.best_index_\n'
                   'scores = np.array([search.cv_results_[f\'split{i}_test_score\'][best_index]\n'
                   f'                   for i in range({num_folds})])\n')
        model_code = 'model = search.best_estimator_\n'
        params_code = ('print(\'Best parameters:\')\n'
                       'for p in params:\n'
                       '    print(f\"- {p[p.index(\'__\')+2:]}: {search.best_params_[p]}\")\n')
    
    code = '{}\n\n'.format('\n'.join(import_code))
    code += f'{xy_code}'
    code += f'cv = KFold(n_splits={num_folds}, shuffle=True, random_state=0)\n'
    code += f'{cv_code}'
    code += f'{model_code}\n'
    code += f'{params_code}'
    code += 'print(f\'Cross-validation score: {scores.mean().round(4)}\')\n'
    code += f'{test_code}'
    code += f'{print_test_code}'
    code += f'\nindex=[f\'fold{{i}}\' for i in range({num_folds})]\n'
    code += 'table = pd.DataFrame({\'R-squared\': scores.round(4)}, index=index).T\n'
    code += 'print(f\'\\n{table}\')\n'
    
    return f'```python\n{code}```'


def pred_reg_analysis_code(ptype, test_switch):

    import_code = ['from sklearn.model_selection import cross_val_predict']
    if ptype:
        import_code.append('import matplotlib.pyplot as plt')
        fig_code = 'plt.figure(figsize=(4.5, 4.5))\n'
    else:
        ptype = []
        fig_code = ''
    if len(test_switch) > 0:
        ytest_code = ('model.fit(x_train, y_train)\n'
                      'yhat_test = model.predict(x_test)\n')
        pred_test_code = ('plt.scatter(yhat_test, y_test,\n'
                          '            facecolor=\'none\', edgecolor=\'red\', label=\'Test\')\n')
        resid_test_code = ('resid_test = y_test - yhat_test\n'
                           'plt.scatter(yhat_test, resid_test,\n'
                           '            facecolor=\'none\', edgecolor=\'red\', label=\'Test\')\n')
        minmax_code = 'ymin, ymax = min([ymin, y_test.min()]), max([ymax, y_test.max()])\n'
        legend_code = 'plt.legend(bbox_to_anchor=(1.02, 1), loc=\'upper left\')\n'
    else:
        ytest_code = ''
        pred_test_code = ''
        resid_test_code = ''
        minmax_code = ''
        legend_code = ''
    
    if 'prediction plot' in ptype:
        pred_code = (f'\n{fig_code}'
                     'plt.scatter(yhat_cv, y_train,\n'
                     '            facecolor=\'none\', edgecolor=\'blue\',\n'
                     '            label=\'Cross-validation\')\n'
                     f'{pred_test_code}'
                     'ymin, ymax = min([yhat_cv.min(), y.min()]), max([yhat_cv.max(), y.max()])\n'
                     f'{minmax_code}'
                     'plt.plot([ymin, ymax], [ymin, ymax], color=\'k\', linestyle=\'--\')\n'
                     f'{legend_code}'
                     'plt.xlabel(\'Predicted values\')\n'
                     'plt.ylabel(\'Actual values\')\n'
                     'plt.grid()\n'
                     'plt.show()\n')
    else:
        pred_code = ''
    if 'residual plot' in ptype:
        resid_code = (f'\n{fig_code}'
                      'resid_cv = y_train - yhat_cv\n'
                      'plt.scatter(yhat_cv, resid_cv,\n'
                      '            facecolor=\'none\', edgecolor=\'blue\',\n'
                      '            label=\'Cross-validation\')\n'
                      f'{resid_test_code}'
                      'ymin, ymax = yhat_cv.min(), yhat_cv.max()\n'
                      f'{minmax_code}'
                      'plt.plot([ymin, ymax], [0, 0], color=\'k\', linestyle=\'--\')\n'
                      f'{legend_code}'
                      'plt.xlabel(\'Predicted values\')\n'
                      'plt.ylabel(\'Actual values\')\n'
                      'plt.grid()\n'
                      'plt.show()\n')
    else:
        resid_code = ''

    code = '{}\n\n'.format('\n'.join(import_code))
    code += 'yhat_cv = cross_val_predict(model, x_train, y_train, cv=cv)\n'
    code += f'{ytest_code}'
    code += f'{pred_code}'
    code += f'{resid_code}'
    
    return f'```python\n{code}```'


def pred_cls_analysis_code(classes_values, cat, test_switch, threshold):

    import_code = ['from sklearn.model_selection import cross_val_predict',
                   'from sklearn.metrics import roc_curve, confusion_matrix',
                   'import matplotlib.pyplot as plt']

    cv_code = ('proba_cv = pd.DataFrame(cross_val_predict(model, x_train, y_train,\n'
               '                                          cv=cv, method=\'predict_proba\'),\n'
               f'                        columns={list(classes_values)})\n')
    cats = cat.__repr__()
    cv_curve = (f'fpr, tpr, thrds = roc_curve(y_train=={cats}, proba_cv[{cats}])\n'
                'plt.plot(fpr, tpr, color=\'b\', label=\'Cross-validation\')\n'
                'plt.scatter(cmat_cv.iloc[0, 1], cmat_cv.iloc[1, 1], s=80, color=\'b\', alpha=0.6)\n'
                'plt.plot([0, 1], [0, 1], color=\'k\', linestyle=\'--\')\n')
    
    normal_code = 'normalize=\'true\''
    index_code = 'index=[\'Actual false\', \'Actual true\']'
    columns_code = 'columns=[\'Predicted false\', \'Predicted true\']'
    if isinstance(cat, np.bool_):
        ycv = 'y_train' if cat else '~y_train'
        ytest = 'y_test' if cat else '~y_test'
    else:
        ycv = f'y_train=={cats}'
        ytest = f'y_test=={cats}'
    cv_cmat = (f'ypred_cv = proba_cv[{cats}] >= threshold\n'
               f'cmat_cv = pd.DataFrame(confusion_matrix({ycv}, ypred_cv, {normal_code}),\n'
               f'                       {index_code},\n'
               f'                       {columns_code}).round(4)\n'
               'print(f\'\\nCross-validation\\n{cmat_cv}\')\n')
    if len(test_switch) > 0:
        test_code = ('model.fit(x_train, y_train)\n'
                     'proba_test = pd.DataFrame(model.predict_proba(x_test),\n'
                     f'                          columns={list(classes_values)})\n')
        test_curve = (f'fpr, tpr, thrds = roc_curve(y_test=={cats}, proba_test[{cats}])\n'
                      'plt.plot(fpr, tpr, color=\'r\', label=\'Test\')\n')
        test_cmat = (f'ypred_test = proba_test[{cats}] >= threshold\n'
                     f'cmat_test = pd.DataFrame(confusion_matrix({ytest}, ypred_test, {normal_code}),\n'
                     f'                         {index_code},\n'
                     f'                         {columns_code}).round(4)\n'
                     'print(f\'\\nTest\\n{cmat_test}\')\n')
    else:
        test_code = ''
        test_curve = ''
        test_cmat = ''
    
    code = '{}\n\n'.format('\n'.join(import_code))
    code += f'{cv_code}'
    code += f'{test_code}'
    code += f'threshold = {threshold}\n'
    code += 'print(f\'threshold: {threshold}\')\n'
    code += f'{cv_cmat}'
    code += f'{test_cmat}'
    code += 'plt.figure(figsize=(4.5, 4.5))\n'
    code += f'{cv_curve}'
    code += f'{test_curve}'
    code += ('plt.legend(loc=\'lower right\')\n'
            'plt.xlabel(\'False positive rate\')\n'
            'plt.ylabel(\'True positive rate\')\n'
            'plt.grid()\n'
            'plt.show()\n')
    
    return f'```python\n{code}```'
