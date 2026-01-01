from shiny.express import ui


def doc_html(docs, margin="4px"):

    line, fields = docs

    blist = [f"<li style='margin-bottom:{margin};margin-left:-15px'><code>{key}</code>: {value}</li>"
             for key, value in fields.items()]
    bullets = (
        "<ul style='margin-bottom:2px'>\n"
        f"{'\n'.join(blist)}\n"
        "</ul>"
    )

    return (
        f"<p style='margin-bottom:{margin}'>{line}</p>\n"
        f"{bullets}"
    )

question_circle_fill = ui.tags.img(src="question-circle-fill.svg",
                                   alt="My Local Icon", style="width:20px;padding-bottom:5px")

ops_menu_dict = {
    "Value counts operations": (
        "Return a dataset summarizing the counts or proportions of unique values.",
        {
            "Columns": "Select one or more columns to count unique values. ",
            "Unstack levels": "Select columns whose levels are unstacked into columns.",
            "Sort": "Sort counts or proportions of values, if it is switched on.",
            "Descending": (
                "Switch between ascending (off) and descending (on) order, "
                "if <code>Sort</code> is switched on."
            ),
            "Normalize": "Switch between counts (off) and proportions (on).",
            "Reset index": (
                "Turn the row indexes into a new column of the output dataset, "
                "and reset the row indexes as default integers, if it is switched on"
            )
        }
    ),
    "Select columns": (
        "Return a dataset of selected columns.",
        {
            "Columns": "Select columns to construct a subset of data."
        }
    ),
    "Sort rows": (
        "Return a dataset with rows sorted by the given columns.",
        {
            "Sort on columns": "Select one or more columns to sort by.",
            "Descending": "Switch between ascending (off) and descending (on) order.",
            "Reset index": "Reset row indices after sorting, if it is switched on."
        }
    ),
    "Boolean conditions": (
        "Return a dataset with filtered rows or a column of boolean values based on logic conditions.",
        {
            "Target variable": "Column to construct a boolean expression.",
            "Operator": (
                "Available operator for constructing the boolean expression "
                "(<code>==</code>, <code>!=</code>, <code><</code>, <code>></code>, etc)."
            ),
            "Value(s) to compare": "The comparison value(s) for the expression.",
            "New bool": "Add the current boolean expression to the logic condition.",
            "Filter rows": (
                "Switch between adding a column of boolean values (off) and "
                "filtering a subset of rows (on) based on logic conditions."
            ),
            "Reset index": (
                "Reset row indices after filtering, if it is switched on. "
                "It only appears when <code>Filter rows</code> is switched on."
            ),
            "To column": (
                "Label of the new column to save the resultant boolean values. "
                "It only appears when <code>Filter rows</code> is switched off."
            )
        }
    ),
    "Treat missing values": (
        "Return a dataset where the missing records are dropped or replaced by given values.",
        {
            "Columns": (
                "Columns that contain missing values. "
                "The method is applied to all columns if the selected columns are not specified."
            ),
            "Method": (
                "The method for treating the missing values: "
                "<code>drop</code> for removing rows with missing values; "
                "or <code>fill</code> for replacing the missing records by a given value."
            ),
            "Value to fill": (
                "Value to fill missing entries with. "
                "It only appears when the selected method is <code>fill</code>."
            ),
            "Reset index": (
                "Reset row index after dropping rows, if it is switched on. "
                "It only appears when the selected method is <code>drop</code>"
            )
        }
    ),
    "Correlation": (
        "Return a dataset of correlation or covariance among selected columns, with optional row dropping.",
        {
            "Metric": "Metric to be calculated: correlation or covariance.",
            "Columns": "Columns of numeric values to conduct the calculation.",
            "Drop rows": "Rows to exclude from calculation."
        }
    ),
    "Aggregation": (
        "Return a dataset of aggregation operations (mean, sum, variance, etc) on selected columns.",
        {
            "Columns": "Columns to conduct the aggregation operations.",
            "Methods": "Methods in terms of aggregation functions (mean, sum, variance, etc).",
            "Transpose": "Transpose the returned dataset."
        }
    ),
    "Group by": (
        "Return a dataset of aggregation operations conducted by groups defined by one or more columns.",
        {
            "Group by": "Columns to split the groups.",
            "View on": "Columns of values to apply the selected aggregation functions.",
            "Methods": "Methods in terms of aggregation functions (mean, sum, variance, etc).",
            "Reset index": (
                "Turn the row indexes into a new column of the output dataset, "
                "and reset the row indexes as default integers, if it is switched on"
            ),
            "Transpose": "Transpose the returned dataset."
        }
    ),
    "Pivot table": (
        (
            "Return a pivot table where the aggregation operations are applied to selected columns "
            "respectively for groups specified as row indexes and column labels."
        ),
        {
            "View on": "Values to aggregate.",
            "Row index": "Columns to split the groups used as rows of the pivot table.",
            "Columns": "Columns to split the groups and used as columns of the pivot table.",
            "Methods": "Methods in terms of aggregation functions (mean, sum, variance, etc).",
            "Reset index": (
                "Turn the row indexes into a new column of the output dataset, "
                "and reset the row indexes as default integers, if it is switched on"
            ),
            "Transpose": "Transpose the returned dataset."
        }
    ),
    "Add columns": (
        "Return a dataset with new columns created according to selected operations.",
        {
            "Expression type": "Type of transformation (arithmetic, type conversion, string operations, etc).",
            "From column(s)": "Source column(s) for transformation.",
            "To column(s)": "Label of the new column(s).",
            "Formula": (
                "The formula of the arithmetic operation. "
                "It only appears when the selected expression type is <code>Arithmetic expression</code>."
            ),
            "Data type": (
                "The data type for conducting data type conversions. "
                "It only appears when the selected expression type is <code>Type conversion</code>."
            ),
            "Method": (
                "The method of vectorized string operations. "
                "It only appears when the selected expression type is <code>String operations</code>."
            ),
            "Drop first": (
                "Drop the dummy variable for the first (reference) category, "
                "if it is switched on. "
                "It only appears when the selected expression type is <code>To dummies</code>."
            ),
            "To binaries": (
                "Convert the boolean type dummy variables into binary integers (0 or 1)."
                "It only appears when the selected expression type is <code>To dummies</code>."
            ),
            "Bins": (
                "The number of bins, or the values of bin edges for creating segments. "
                "It only appears when the selected expression type is <code>To dummies</code>."
            ),
            "Labels": (
                "Labels of segmented bins. The number of labels must match the number of bins. "
                "It only appears when the selected expression type is <code>To segments</code>."
            ),
        }
    ),
    "Date time": (
        "Return a dataset with new columns created based on the derived date/time/duration information",
        {
            "Column": "Column to derive the date/time/duration information.",
            "Format": (
                "A string specifying the date/time format. "
                "It only works when the selected column is a series of strings."
            ),
            "To duration": (
                "Compute duration between the selected column and a reference time, "
                "if it is switched on. "
                "It only appears when the selected column is not a duration."
            ),
            "Calendar/Hour/Minute/Second": (
                "Date and clock values of the reference time for calculating the duration. "
                "They only appear when <code>To duration</code> is switched on."
            ),
            "Reverse": "Reserve the duration values if it is switched on.",
            "Results": "Results of the time or duration information (year, month, day, etc).",
            "Label prefix": "Prefix for the labels of new columns."
        }
    ),
    "Time trend": (
        "Return a dataset with new columns in terms of trend analysis.",
        {
            "Select all": "Select all numeric columns.",
            "Columns": "Columns to conduct trend analysis.",
            "Transform": "Type of time trend or rolling window calculation.",
            "Steps": "Lag or window size. It can be specified as multiple integers separated by commas.",
            "Drop original data": "Remove original columns after transformation."
        }
    ),
    "Clustering": (
        "Return a dataset with new columns of cluster labels created using K-means or hierarchical clustering.",
        {
            "Select all": "Select all eligible columns.",
            "Drop NA": "Drop rows with missing values if it is switched on.",
            "Features for clustering": "Columns representing the features to conduct clustering.",
            "Numbers treated as categories": "Numeric columns treated as categorical features.",
            "Method": "Clustering algorithm: K-means or hierarchical clustering.",
            "Cluster No.": "Number of clusters. It can be specified as multiple integers separated by commas.",
            "Label prefix": "Prefix for the new columns representing the clustering results.",
            "Value prefix": "Prefix for the labels of clusters.",
        }
    ),
    "Decomposition": (
        "Return a dataset with new columns of features as a result of decomposition.",
        {
            "Select all": "Select all eligible columns.",
            "Drop NA": "Drop rows with missing values if it is switched on.",
            "Features for decomposition": "Columns representing the features to conduct decomposition.",
            "Numbers treated as categories": "Numeric columns treated as categorical features.",
            "Scaling": "Scaling method applied to features: not applied, standardize, or normalize.",
            "Method": "Decomposition method: PCA, kernel PCA, NMF, or factor analysis. ",
            "Kernel": (
                "Kernel type: linear, polynomial, RBF, sigmoid, or cosine. "
                "It only appears when the selected <code>Method</code> is kernel PCA (<code>KernelPCA</code>)."
            ),
            "Degree": (
                "Polynomial kernel degree. "
                "It only appears when the selected <code>Kernel</code> is polynomial (<code>poly</code>)."
            ),
            "Label prefix": "Prefix for the new columns representing the features after decomposition.",
            "Show first": "Number of features after decomposition to be included in the returned dataset.",
            "Replace original features": "Drop selected columns after decomposition."
        }
    ),
    "ANOVA": (
        "Return a dataset summarizing the results of analysis of variance (ANOVA) for group comparisons.",
        {
            "Numerical target": "Dependent (response) variable.",
            "Features": "Grouping (independent) variables.",
            "Formula": "Model formula.",
            "Type": "ANOVA type (I, II, III).",
            "Test": "Test statistic (F, Chisq, Cp)."
        }
    ),
    "Variance inflation factor": (
        "Return a dataset summarizing the results of VIF for assessing multicollinearity among features.",
        {
            "Features": "Columns representing the features to compute VIF for.",
            "Intercept": "Include constant term if it is switched on.",
            "Reset index": (
                "Turn the row indexes into a new column of the output dataset, "
                "and reset the row indexes as default integers, if it is switched on"
            )
        }
    ),
    "Random sampling": (
        "Return a dataset of randomly sampled rows (with or without replacement) in a number of batches.",
        {
            "Select all": "Select all columns.",
            "Drop NA": "Drop rows with missing data.",
            "Columns": "Columns to include in sampling.",
            "Sample size": "Number of rows per batch.",
            "Batch number": "Number of batches.",
            "Random state": "Random seed.",
            "Replace": "Switch between sampling without replacement (off) and with replacement (on).",
            "Reset index": "Reset index after sampling if it is switched on."
        }
    ),
    "Over sampling": (
        "Return a dataset as the result of synthetic over-sampling for achieving data balance.",
        {
            "Categorical target": "Column representing the target categorical variable.",
            "Features": "Columns of features involved in oversampling.",
            "Method": "Oversampling algorithm: random sampling, SMOTE, or ADASYN.",
            "Strategy": "Resampling strategy.",
            "Neighbor No.": "Number of nearest neighbors (for SMOTE/ADASYN)."
        }
    ),
}

dvs_menu_dict = {
    "Value counts": (
        (
            "Plot a bar chart that visualizes the distribution of discrete values "
            "in terms of counts or normalized proportions."
        ),
        {
            "Column": "Column selected to visualize the distribution.",
            "Direction": "The direction of bar chart: vertical or horizontal.",
            "Method": "The method for visualizing the distribution: in actual counts or proportions.",
            "Palette": "Color selected for the bar chart.",
            "Opacity": "Opacity of the bar chart."
        }
    ),
    "Probability plot": (
        "Create a Q-Q plot for assessing if a numeric data column follows a given theoretical distribution.",
        {
            "Column": "Column of a numeric variable selected to visualize the distribution.",
            "Distribution": "Theoretical distribution to compare (Normal, Exponential, Uniform).",
            "Standardize": "Standardize data before plotting if it is switched on.",
            "Palette": "Color selected for markers.",
            "Opacity": "Opacity of markers."
        }
    ),
    "Histogram": (
        (
            "Plot histograms that visualize the distribution of a numeric data column "
            "with optional grouping and styling options."
        ),
        {
            "Column": "Column of a numeric variable selected to visualize the distribution.",
            "Group": "Column for creating groups of histograms with different colors.",
            "Bins": "Number of histogram bins.",
            "Method": "The method for visualizing the distribution: count or density.",
            "Palette": (
                "Color selected for the histogram. "
                "It only appears when <code>Group</code> is not specified."
            ),
            "Theme": (
                "The color theme (color map) of groups of histograms with different colors. "
                "It only appears when <code>Group</code> is specified."
            ),
            "Normalized": (
                "The method for normalizing groups of histograms: separately or jointly. "
                "It only appears when <code>Group</code> is specified."
            ),
            "Style": (
                "The style of arranging groups of histograms: layer, stack, or fill."
                "It only appears when <code>Group</code> is specified."
            ),
            "Opacity": "Opacity of histograms."
        }
    ),
    "KDE": (
        (
            "Create kernel density estimate (KDE) plot for a numeric data column. "
            "with optional grouping and styling options."
        ),
        {
            "Column": "Column of a numeric variable selected to visualize the distribution.",
            "Group": "Column for creating groups of KDE plots with different colors.",
            "Palette": (
                "Color selected for the KDE. "
                "It only appears when <code>Group</code> is not specified."
            ),
            "Theme": (
                "The color theme (color map) of groups of KDE plots with different colors. "
                "It only appears when <code>Group</code> is specified."
            ),
            "Normalized": (
                "The method for normalizing groups of KDE plots: separately or jointly. "
                "It only appears when <code>Group</code> is specified."
            ),
            "Style": (
                "The style of arranging groups of KDE plots: layer, stack, or fill."
                "It only appears when <code>Group</code> is specified."
            ),
            "Opacity": "Opacity of KDE plots."
        }
    ),
    "Box plot": (
        "Create box plot for a numeric data column with optional grouping and styling options.",
        {
            "Column": "Column of a numeric variable selected to visualize the distribution.",
            "Group": "Column for creating groups of box plots.",
            "Hues": "Column for creating various colors within each group.",
            "Notch": "Display notch for sample median if it is switched.",
            "Mean": "Show mean value as a black marker if it is switched.",
            "Direction": "The direction of box plots: vertical or horizontal.",
            "Width": "Width of each group of box plots.",
            "Palette": (
                "Color selected for the boxplot. "
                "It only appears when <code>Hues</code> is not specified."
            ),
            "Theme": (
                "The color theme (color map) of box plots with different colors "
                "to indicate the values of <code>Hues</code>. "
                "It only appears when <code>Hues</code> is specified."
            ),
            "Opacity": "Opacity of box plots."
        }
    ),
    "Pair plot": (
        "Creates a grid of scatter plots and histograms (or KDEs) for pairwise relations among selected columns.",
        {
            "Columns": "Columns of numeric variables selected to create the pair plot.",
            "Drop rows": "Rows to exclude from the pair plot.",
            "Hues": "Column for creating various colors of scatter plots and histograms (or KDEs).",
            "Theme": (
                "The color theme (color map) of scatter plots and histograms "
                "to indicate the values of <code>Hues</code>. "
                "It only appears when <code>Hues</code> is specified."
            ),
            "Plot kind": "Type of off-diagonal plots (scatter, kde, hist, reg).",
            "Diagonal kind": "Type of diagonal plots (auto, kde, hist).",
            "Corner": "Display only lower triangle.",
            "Opacity": "Plot opacity."
        }
    ),
    "Heat map": (
        "Plot the heat map for a dataset of numeric values.",
        {
            "Columns": "Columns of numeric variables selected to plot the heat map.",
            "Theme": "The color theme (color map) of the heat map.",
            "Annotate": "Show numeric values in cells if it is switched on.",
            "Ticks at top": (
                "Switch between displaying the column labels "
                "at bottom (off) or top (on) of the heat map."
            )
        }
    ),
    "Bar chart": (
        "Plot a bar chart visualizing a single or multiple series.",
        {
            "Y-data": "Column of a numeric variable selected for one group of bars.",
            "Label": (
                "Label of the current bar. "
                "The default value is the selected column label."
            ),
            "Palette": "Color selected for the current bar.",
            "New bar": "Add the current bar to the bar chart.",
            "Sort bars": "Sort the bars if it is switched on.",
            "Descending": (
                "Switch between ascending (off) and descending (on) order. "
                "It only appears when <code>Sort bars</code> is switched on."
            ),
            "Sort by": (
                "Column selected to sort bars by. "
                "It only appears when <code>Sort bars</code> is switched on."
            ),
            "X-data": (
                "Column selected for the X-data of all bars. "
                "If unspecified, row indexes will be used as X-data."
            ),
            "Direction": "The direction of the bar chart: vertical or horizontal.",
            "Style": "The style of arranging bars: clustered or stack.",
            "Width": "Width of each group of bars.",
            "Opacity": "Opacity of bars."
        }
    ),
    "Radar chart": (
        "Plot radar chart for displaying multivariate data in polar coordinates.",
        {
            "Columns": "Columns of numeric variables selected to plot the radar chart.",
            "Category": "Column selected as the angular tick labels of the radar chart.",
            "Tick axis angle": "Angle for radial tick labels.",
            "Theme": "The color theme (color map) of the heat map.",
            "Opacity": "Opacity of filled areas."
        }
    ),
    "Line plot": (
        "Create line plots for one or multiple data series, with optional error margins and styling.",
        {
            "Y-data": "Column of a numeric variable selected as the Y-data of the current line plot.",
            "X-data": (
                "Column selected as the X-data of the current line plot. "
                "If unspecified, row indexes will be used as X-data."
            ),
            "Margin": (
                "Columns for deviations from the current line, displayed as shaded areas. "
                "If one column is selected, values of the column is used to indicate "
                "both the downward and upward deviations, so the displayed shaded areas are symmetrical. "
                "If two columns are selected, they are used to specify the downward and upward deviations, "
                "respectively."
            ),
            "Label": (
                "Label of the current line. "
                "The default value is the selected column label."
            ),
            "Style": "Line style (solid, dash, dot, etc).",
            "Marker": "Marker type.",
            "Width": "Line width.",
            "Scale": "Marker size scaling.",
            "Palette": "Color selected for the current line.",
            "New bar": "Add the current line to the line plot.",
        }
    ),
    "Filled areas": (
        "Plot filled areas for selected data series.",
        {
            "Columns": "Columns of numeric variables selected as the Y-data of the filled areas.",
            "X-data": (
                "Column selected as the X-data of the filled areas. "
                "If unspecified, row indexes will be used as X-data."
            ),
            "Style": "Style of arranging the filled areas: layer or stack.",
            "Theme": "The color theme (color map) of filled colors.",
            "Opacity": "Opacity of all filled areas."
        }
    ),
    "Scatter plot": (
        "Create a scatter plot with varying colors and marker sizes.",
        {
            "Y-data": "Column selected as the Y-data of the scatter plot.",
            "X-data": (
                "Column selected as the X-data of the current line plot. "
                "If unspecified, row indexes will be used as X-data."
            ),
            "Size": "Column of a numeric variable to represent marker sizes.",
            "Scale": "Marker size scaling.",
            "Hues": "Column for creating various colors of markers.",
            "Palette": (
                "Color selected for the scatter plot. "
                "It only appears when <code>Hues</code> is not specified."
            ),
            "Theme": (
                "The color theme (color map) of the markers with different colors "
                "to indicate the values of <code>Hues</code>. "
                "It only appears when <code>Hues</code> is specified."
            ),
            "Opacity": "Marker opacity."
        }
    ),
    "Regression plot": (
        (
            "Create a scatter plot with a fitted regression curve and optional confidence intervals "
            "generated from bootstrap."
        ),
        {
            "Y-data": "Column of a numeric variable selected as the Y-data of the regression plot.",
            "X-data": "Column of a numeric variable selected as the X-data of the regression plot",
            "Hues": "Column for creating various colors of markers and fitted lines.",
            "Palette": (
                "Color selected for the scatter plot and fitted lines. "
                "It only appears when <code>Hues</code> is not specified."
            ),
            "Theme": (
                "The color theme (color map) of the markers with different colors "
                "to indicate the values of <code>Hues</code>. "
                "It only appears when <code>Hues</code> is specified."
            ),
            "Opacity": "Marker opacity.",
            "Fitted line": "Display the fitted regression curve if it is switched on.",
            "Centroid": "Display centroids of markers with different colors.",
            "Confidence level": "Confidence interval for fit.",
            "Transformation": (
                "Apply transformation on X-data: "
                "polynomial, log, or logistic."
            ),
            "Polynomial order": (
                "Order for polynomial fit. "
                "It only appears when <code>Transformation</code> is specified to be <code>poly</code>."
            )
        }
    ),
    "ACF and PACF": (
        "Create autocorrelation (ACF) or partial autocorrelation (PACF) plots for selected time series.",
        {
            "Columns": "Column of a numeric variable selected to create the ACF or PACF plots.",
            "Plot type": "Type of plot: ACF or PACF.",
            "Method": "Estimation method.",
            "Lags": "Maximum number of lags to plot.",
            "Confidence level": "Plot confidence intervals if it is switched on.",
            "Palette": "Color of the ACF or PACF stem plot."
        }
    ),
}

mds_menu_dict = {
    "Statsmodels":(
        "Fit a model created using the <code>statsmodels</code> package.",
        {
            "Dependent variable": "Dependent (explained) variable of the model.",
            "Independent variables": "Independent (explanatory) variables of the model.",
            "Numbers treated as categories": "Columns of numeric predictors treated as categorical variables.",
            "Formula": (
                "A string for specifying model formula. "
                "The formula is specified according to the syntax rules of the <code>dmatrix</code> package."
            ),
            "Model type": (
                "Type of model used to fit the data: "
                "ordinary least square (ols) or logistic regression (logit)."
            ),
            "Output name": (
                "The name of the output dataset. "
                "The dataset is used to summarize the information on model coefficients, "
                "in terms of parameter values, standard error, t-values, p-values, and confidence intervals."
            ),
            "Fit model": "Fit the model with the current formula and settings."
        }
    ),
    "Scikit-learn models": (
        "Specify and fit a model using the <code>scikit-learn</code> package.",
        dict()
    )
}

sklearn_page_dict = [
    (
        (
            "Specify the predicted and predictor variables of model. "
            "You may also define the model formula in terms of nonlinear or interaction terms, "
            "and how the categorical variables are converted into dummy variables."
        ),
        {
            "Dependent variable": "Dependent (predicted) variable of the model.",
            "Independent variables": "Independent (predictor) variables of the model.",
            "Numbers treated as categories": "Columns of numeric predictors treated as categorical variables.",
            "Edit Formula": "Enable editing model formula if it is switched on. ",
            "Formula": (
                "A string for specifying model formula. "
                "The formula is specified according to the syntax rules of the <code>dmatrix</code> package. "
                "It only appears when the <code>Edit Formula</code> is switched off."
            ),
            "Drop first": (
                "Drop the dummy variable for the first (reference) category if it is switched on. "
                "It only appears when the <code>Edit Formula</code> is switched off."
            )
        }
    ),
    (
        "Specify the pipeline of data preprocessing steps and the predictive model.",
        {
            "Log transformation of response": (
                "Log transformation of the response variable before being used for . "
                "training a regression model. "
                "It only appears when the response is a numeric variable."
            ),
            "Over-sampling of response": (
                "Creating balanced response observations via various over-sampling approaches: "
                "random over-sampling, SMOTE, or ADASYN."
            ),
            "Scaling": (
                "Scaling method of predictor variables: "
                "not applied, standardization, or normalization."
            ),
            "PCA": (
                "Reduced number of features created by principle component analysis (PCA). "
                "It can be specified as a sequence of integers for grid search. "
                "PCA is not applied if <code>PCA</code> is not specified."
            ),
            "Model selection": "Select a model for regression or classification.",
            "<i>optional hyper-parameters</i>": (
                "Hyper-parameters associated with the selected predictive model. "
                "Each parameter can be a singular value or "
                "a collection of candidate values for grid search. "
                "If they hyper-parameter is not specified, the default value will be applied."
            )
        }
    ),
    (
        (
            "Fit and test the model with the specified validation settings. "
            "If multiple hyper-parameter values are defined in the previous step, "
            "grid search will be applied to find the hyper-parameters with the best model performance."
        ),
        {
            "CV Folds": "The number of folds for cross-validation.",
            "Test ratio": (
                "A test set will be split from the overall dataset if it is switched on. "
                "You may then specify the ratio of the test dataset."
            ),
            "Fit model": "Fit the model with the current training and validation settings."
        }
    ),
    (
        "Output a dataset with the predicted responses and selected visualizations.",
        {
            "Predict plot": (
                "Output the predict plot if it is checked. "
                "It only appears when the predicted variable is a numeric response."
            ),
            "Residual plot": (
                "Output the residual plot if it is checked. "
                "It only appears when the predicted variable is a numeric response."
            ),
            "Confusion matrix": (
                "Output the cross-validation and test confusion matrix. "
                "It only appears when the predicted variable is a categorical response."
            ),
            "Receiver-operating characteristic": (
                "Output the receiver-operating characteristic curve (ROC). "
                "It only appears when the predicted variable is a categorical response "
                "and <code>Target class</code> is specified."
            ),
            "Precision-recall": (
                "Output the precision-recall curve. "
                "It only appears when the predicted variable is a categorical response "
                "and <code>Target class</code> is specified."
            ),
            "Feature importance": (
                "Output the feature importance plot. "
                "It only appears when the selected model supports feature importance, "
                "such as decision tree or tree ensembles."
            ),
            "Target class": (
                "A target class used for plotting the ROC and precision-recall curves. "
                "It only appears when the predicted variable is a categorical response."
            ),
            "Output name": (
                "The name of the output dataset. "
                "In the output dataset, the predictions and validation split labels "
                "are included as new columns."
            )
        }
    )
]

op_cats = {"Data selection and treatment": ["Select columns", "Boolean conditions", "Sort rows",
                                            "Treat missing values", "Add columns"],
           "Aggregation and statistics": ["Value counts operations",
                                          "Aggregation", "Group by", "Pivot table",
                                          "Correlation", "ANOVA", 
                                          "Variance inflation factor", "Decomposition", "Clustering"],
           "Time series": ["Date time", "Time trend"],
           "Data sampling": ["Random sampling", "Over sampling"]}

dv_cats = {"Distribution plots": ["Value counts", "Histogram", "KDE", "Box plot", "Probability plot"],
           "Relation and correlation": ["Pair plot", "Scatter plot", "Regression plot"],
           "Value comparison": ["Bar chart", "Radar chart", "Heat map"],
           "Trend and time series": ["Line plot", "Filled areas", "ACF and PACF"]}

md_cats = {"Predictive analytics": ["Statsmodels", "Scikit-learn models"]}
