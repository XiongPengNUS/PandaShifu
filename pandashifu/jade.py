def export(nodes):

    import_map = {'sns': 'import seaborn as sns',
                  'plt': 'import matplotlib.pyplot as plt',
                  'smf': 'import statsmodels.formula.api as smf',
                  'pd': 'import pandas as pd',
                  'np': 'import numpy as np'}
    
    from_map = {'Pipeline': 'from sklearn.pipeline import Pipeline',
                'OneHotEncoder': 'from sklearn.preprocessing import OneHotEncoder',
                'StandardScaler': 'from sklearn.preprocessing import StandardScaler',
                'Normalizer': 'from sklearn.preprocessing import Normalizer',
                'PCA': 'from sklearn.decomposition import PCA',
                'DecisionTreeRegressor': 'from sklearn.tree import DecisionTreeRegressor',
                'DecisionTreeClassifier': 'from sklearn.tree import DecisionTreeClassifier',
                'RandomForestRegressor': 'from sklearn.ensemble import RandomForestRegressor',
                'RandomForestClassifier': 'from sklearn.ensemble import RandomForestClassifier',
                'ColumnTransformer': 'from sklearn.compose import ColumnTransformer',
                'LinearRegression': 'from sklearn.linear_model import LinearRegression',
                'Lasso': 'from sklearn.linear_model import Lasso',
                'Ridge': 'from sklearn.linear_model import Ridge',
                'LogisticRegression': 'from sklearn.linear_model import LogisticRegression',
                'KFold': 'from sklearn.model_selection import KFold',
                'train_test_split': 'from sklearn.model_selection import train_test_split',
                'cross_val_score': 'from sklearn.model_selection import cross_val_score',
                'cross_val_predict': 'from sklearn.model_selection import cross_val_predict',
                'roc_curve': 'from sklearn.metrics import roc_curve, confusion_matrix',
                'GridSearchCV': 'from sklearn.model_selection import GridSearchCV'}

    data_cell = dict(cell_type='code', metadata={}, source='# read the dataset \'data\'')

    code_cells = [data_cell]
    all_imports = set()
    for node in nodes[1:]:
        code_string = node.code
        for key in import_map:
            if f'as {key}' in node.code:
                all_imports.add(key)
                code_string = code_string.replace(f'{import_map[key]}\n\n', '')
                code_string = code_string.replace(f'{import_map[key]}\n', '')
        for key in from_map:
            if f'import {key}' in node.code:
                all_imports.add(key)
                code_string = code_string.replace(f'{from_map[key]}\n\n', '')
                code_string = code_string.replace(f'{from_map[key]}\n', '')

        if node.ntype == 'data':
            output = f'\n{node.content[0]}'
        elif node.ntype == 'visual' or node.ntype == 'model':
            output = ''

        code_dict = dict(cell_type='code',
                         metadata={},
                         source=f'{code_string}{output}')
        code_cells.append(code_dict)
    
    imports = ['import pandas as pd',
               'import numpy as np']
    for item in all_imports:
        if item in import_map and item not in ['pd', 'np']:
            imports.insert(2, import_map[item])
        if item in from_map:
            imports.append(from_map[item])
    import_cell = dict(cell_type='code', metadata={}, source='\n'.join(imports))
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
