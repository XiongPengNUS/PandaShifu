import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom, norm, t, linregress, multivariate_normal

from shiny import reactive
from shiny.express import render, ui, input

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error as rmse


chd_style = 'color:white; background:#007bc2 !important;'


def ui_block(string, btype):

    return ui.markdown(f'<div class="alert alert-block alert-{btype}">\n{string}</div>')


def ci_equations(param, stats):

    if param == 'mean':
        center = '\\bar{x}'
        center_eq = '=\\frac{1}{n}\\sum_{i=1}^nx_i'
        if stats == 'sigma':
            se = '\\frac{\\sigma}{\\sqrt{n}}' 
            se_text = '<li>Population standard deviation $\\sigma$</li>\n'
        else:
            se = '\\frac{s}{\\sqrt{n}}'
            se_text = '<li>Sample standard deviation $s$.</li>\n'
    else:
        center = '\\hat{p}'
        center_eq = ''
        se = '\\sqrt{\\frac{\\hat{p}(1-\\hat{p})}{n}}'
        se_text = ''
    if stats == 'sigma':
        ppf = 'z_{\\alpha/2}'
        distr, dof = 'standard normal ', ''
    else:
        ppf = 't_{\\alpha/2, n-1}'
        distr, dof = '$t$-', ' with the degree of freedom to be $n-1$'

    return ('$$\n'
            f'{center} \\pm {ppf} \\cdot {se}\n'
            '$$\n'
            '<ul style="margin-bottom: 0px;">'
            f'<li>Sample {param}: ${center}{center_eq}$.</li>\n'
            f'<li>Cut-off value ${ppf}$ as the $(1-\\alpha/2)$th percentile of the {distr}'
            f'distribution{dof}.</li>\n'
            f'{se_text}'
            '<li>Sample size $n$.</li>\n'
            '</ul>')


def mysterious_prf(x):

    return ((1.2 - 0.2*x) * np.sin(9*x + 0.8*x**0.5) + 4*x) * 4


def mysterious(size):

    x = np.random.rand(size)
    #y = ((1.2 - 0.2*x) * np.sin(11*x) + 4*x) * 4 + np.random.randn(size)
    y = mysterious_prf(x) + 2.5*np.random.randn(size)

    return pd.DataFrame({'y': y, 'x': x})

def reg_pred(data, model, params={}):

    x, y = data[['x']], data['y']
    pipe = None
    params = {key:value for key, value in params.items() if value is not None}
    if model == 'Polynomial regression':
        k = params['degree']
        pipe = Pipeline([('poly',  PolynomialFeatures(degree=k, include_bias=False)),
                         ('reg',  LinearRegression(fit_intercept=True))])
    elif model == 'K-nearest neighbors':
        pipe = Pipeline([('Scaler', StandardScaler()),
                         ('reg', KNeighborsRegressor(**params))])
    elif model == 'Decision tree':
        pipe = Pipeline([('reg', DecisionTreeRegressor(random_state=0, **params))])
    elif model == 'Bagged trees':
        pipe = Pipeline([('reg', RandomForestRegressor(random_state=0, n_estimators=80, **params))])
    
    if pipe is not None:
        pipe.fit(x, y)
        x_pred = pd.DataFrame({'x': np.arange(0, 1.01, 0.01)})
        y_pred = pipe.predict(x_pred)

        return x_pred, y_pred, pipe



def two_class_data(n):

    data_dict = {}

    #x = 1 - 3*np.random.rand(2*n, 2)
    x = -0.5 + 0.9*np.random.normal(size=(2*n, 2))
    #x = np.maximum(np.minimum(x, 1.2), -2.2)
    y = np.array([-1]*n + [1]*n)
    x[y==1, 0] += 3.2
    x[y==1, 1] += 3.2
    sep1 = x.sum(axis=1) - 1.8 > 0
    sep2 = x.sum(axis=1) - 2.8 < 0
    x[(y==-1) & sep1, 1] = 1.8 - x[(y==-1) & sep1, 0]
    x[(y==1) & sep2, 0] = 2.8 - x[(y==1) & sep2, 1]
    
    y = np.where(y == 1, 'Class 2', 'Class 1')
    data_dict['Separable'] = pd.DataFrame({'y': y, 'x1': x[:, 0], 'x2': x[:, 1]})


    x = np.random.normal(size=(2*n, 2))
    y = np.array([-1]*n + [1]*n)
    x[y==1, :] += 1.5
    x += 0.3
    y = np.where(y == 1, 'Class 2', 'Class 1')
    data_dict['Barely separable'] = pd.DataFrame({'y': y, 'x1': x[:, 0], 'x2': x[:, 1]})

    x = 0.6*np.random.normal(size=(2*n, 2))
    y = np.array([-1]*n + [1]*n)
    x[:n//2, :] += 3
    x[n//2:n, :] -= 0.5
    x[-n:, :] += 1.5
    y = np.where(y == 1, 'Class 2', 'Class 1')
    data_dict['Non-separable'] = pd.DataFrame({'y': y, 'x1': x[:, 0], 'x2': x[:, 1]})

    return data_dict


def two_class_pred(data, model, params={}):

    x, y = data.drop(columns='y'), data['y']
    pipe = None
    params = {key:value for key, value in params.items() if value is not None}
    if model == 'Logistic regression':
        pipe = Pipeline([('Scaler', StandardScaler()),
                         ('cls', LogisticRegression(tol=1e-6, **params))])
    elif model == 'Linear discriminant analysis':
        pipe = Pipeline([('cls', LinearDiscriminantAnalysis(**params))])
    elif model == 'Quadratic discriminant analysis':
        pipe = Pipeline([('cls', QuadraticDiscriminantAnalysis(**params))])
    elif model == 'Support vector machine':
        pipe = Pipeline([('cls', SVC(**params))])
    elif model == 'K-nearest neighbors':
        pipe = Pipeline([('Scaler', StandardScaler()),
                         ('cls', KNeighborsClassifier(**params))])
    elif model == 'Decision tree':
        pipe = Pipeline([('cls', DecisionTreeClassifier(random_state=0, **params))])
    elif model == 'Bagged trees':
        pipe = Pipeline([('cls', RandomForestClassifier(random_state=0, n_estimators=80,
                                                        max_features=1.0, **params))])
    
    if pipe is not None:
        pipe.fit(x, y)
        xx1, xx2 = np.meshgrid(np.linspace(-4.3, 6.3, 200),
                               np.linspace(-4.3, 6.3, 200))
        dim = len(xx1)
        x_pred = pd.DataFrame(dict(x1=xx1.reshape((xx1.size,)),
                                   x2=xx2.reshape((xx2.size,))))
        if input.class_model() == "Support vector machine":
            proba_pred = pipe.predict(x_pred).reshape(dim, dim) == "Class 1"
        else:
            proba_pred = pipe.predict_proba(x_pred)[:, 0].reshape((dim, dim))

        return xx1, xx2, proba_pred, pipe


math_tag = (
    ui.tags.link(
        rel="stylesheet",
        href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
    ),
    ui.tags.script(src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js"),
    ui.tags.script(src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js"),
    ui.tags.script("""
        document.addEventListener('DOMContentLoaded', function() {
            renderMathInElement(document.body, {
                    delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false},
                ]
            });
        });
    """)
)

with ui.tags.head():
    math_tag

with ui.layout_column_wrap(width="1060px", fixed_width=True):
    with ui.navset_pill_list(id="selected_navset_pill_list", widths=(3, 9), well=False):
        with ui.nav_panel("Probability Theory"):
            ui.markdown('### Discrete Random Variables')
            ui.markdown(('A random variable $X$ is defined to be discrete if its possible '
                        'outcomes are finite or countable. Examples of distributions of '
                        'discrete random variables are discrete uniform distribution (i.e., '
                        'outcome of rolling an even die), Bernoulli distribution (i.e., the '
                        'preference of a randomly selected customer for Coke or Pepsi), '
                        'Binomial distribution (i.e., the number of customers who prefer Coke '
                        'over Pepsi among 10 randomly selected customers), and Poisson '
                        'distribution (i.e., The number of patients arriving in an emergency '
                        'room within a fixed time interval) etc. '))

            ui_block(('<b>Notes: </b> For a discrete random variable $X$ with $k$ possible '
                    'outcomes $x_j$, $j=1, 2, ..., k$, the <b>probability mass function '
                    '(PMF)</b> is given as:\n'
                    '$$\n'
                    '\\begin{align*}\n'
                    'P(X=x_j) = p_j \\text{~~~~for each~}j=1, 2, ..., k, \n'
                    '\\end{align*}'
                    '$$\n'
                    'where $p_j$ is the probability of the outcome $x_j$ and all $p_j$ '
                    'must satisfy \n'
                    '$$\n'
                    '\\begin{cases}\n'
                    '0\\leq p_j \\leq 1 \\text{~~~~for each }j=1, 2, ..., k, \\\\ \n'
                    '\\sum_{j=1}^kp_j = 1. \n'
                    '\\end{cases}\n'
                    '$$\n'), 'danger')

            with ui.card(height='850px'):
                ui.card_header('Binomial distribution', style=chd_style)
                with ui.layout_sidebar():
                    with ui.sidebar(bg='#f8f8f8', width='350px'):  
                        ui.input_slider(id='drv_prop', label='Probability of success:',
                                        min=0.05, max=0.95, value=0.40, step=0.01)
                        ui.input_slider(id='drv_num', label='Number of experiments:',
                                        min=5, max=100, value=25, step=1)
                    
                        @render.ui
                        @reactive.event(input.drv_num)
                        def obs_count():
                            num = input.drv_num()
                            return ui.input_slider(id='drv_m', label=ui.HTML('Number of successes:'),
                                                min=0, max=num, value=num//2, step=0.1)
                        
                        ui.markdown('---')

                        @render.ui
                        @reactive.event(input.drv_prop, input.drv_num, input.drv_m)
                        def update_dvr_code():

                            p = input.drv_prop()
                            n = input.drv_num()
                            m = input.drv_m()
                            pmf = binom.pmf(m, n, p)
                            cdf = binom.cdf(m, n, p)

                            return ui.markdown(('```python\n'
                                                f'binom.pmf({m}, {n}, {p})\n'
                                                '```\n\n'
                                                f'<code>{pmf}</code>\n'
                                                '```python\n'
                                                f'binom.cdf({m}, {n}, {p})\n'
                                                '```\n\n'
                                                f'<code>{cdf}</code>\n'))

                    ui.markdown('The binomial distribution is a discrete probability distribution, '
                                'defined to be the number of successes, denoted by $X$, in a sequence '
                                'of $n$ independent experiments, and each experiment has a Boolean '
                                'valued outcome: success (with probability $p$) or failure (with '
                                'probability $1-p$). The graphs below show the PMF and CDF of such a '
                                'binomial distributed random variable $X$.')

                    @render.plot(width=550, height=600)
                    @reactive.event(input.drv_prop, input.drv_num, input.drv_m)
                    def update_binom_plot():
                        p = input.drv_prop()
                        n = input.drv_num()
                        m = input.drv_m()
                        x1 = np.arange(0, np.floor(m)+1)
                        x2 = np.arange(np.floor(m)+1, n+1)
                        pmf1 = binom.pmf(x1, n, p)
                        pmf2 = binom.pmf(x2, n, p)
                        pmf_m = binom.pmf(m, n, p)
                        xs = np.arange(0, n+0.001, 0.001)
                        cdf = binom.cdf(xs, n, p)
                        cdf_m = binom.cdf(m, n, p)

                        fig, ax = plt.subplots(2, 1)
                        ax[0].vlines(x1, 0, pmf1, linewidth=2, color='b',
                                    label=f'Probability $P(X\\leq{m})$')
                        ax[0].vlines(x2, 0, pmf2, linewidth=2, color='gray',
                                    label=f'Probability $P(X>{m})$')
                        ax[0].scatter(m, pmf_m, s=40, c='r', alpha=0.5,
                                    label=f'$P(X={m}) = {pmf_m.round(4)}$')
                        ax[0].grid()
                        if p > 0.5:
                            ax[0].legend(fontsize=10, loc='upper left')
                        else:
                            ax[0].legend(fontsize=10, loc='upper right')
                        ax[0].set_ylabel('PMF', fontsize=11)
                        ax[1].plot(xs, cdf, color='b', linewidth=1.5)
                        ax[1].scatter(m, cdf_m, s=40, c='r', alpha=0.5,
                                    label=f'$P(X\\leq {m}) = {cdf_m.round(4)}$')
                        ax[1].grid()
                        ax[1].legend(fontsize=10)
                        ax[1].set_ylabel('CDF', fontsize=11)
                        ax[1].set_xlabel('Number of successes', fontsize=11)
                    
                        return fig
            
            ui.markdown('<br>')
            ui.markdown('### Continuous Random Variables')
            ui.markdown(('A variable $X$ is a **continuous random variable** if takes all values in '
                        'an interval of numbers. Random variables following uniform, normal '
                        '(Gaussian) and exponential distributions are all continuous variables. \n'
                        'For continuous random variables, there is no PMF as the discrete random '
                        'variables, because $P(X=x)=0$ for all values of $x$. Only intervals of '
                        'values have positive probabilities, such as $P(0 \\leq x \\leq 10)$. The '
                        'CDF for a continuous random variable has the same definition as the '
                        'discrete case, which is $F(x)=P(X\\leq x)$. Based on the CDF, we have other '
                        'definitions listed as follows.'))
            ui_block(('<b>Notes: </b> Let $F(x) = P(X\\leq x)$ be the CDF of a continuous random '
                    'variable $X$, then\n'
                    '<ul style="margin-bottom: 0px;">'
                    '<li> The derivative $f(x) = \\frac{d F(x)}{dx}$ of the CDF $F(x)$ is called '
                    'the <b>probability density function (PDF)</b> of $X$. This definition also '
                    'implies that $F(x) = \\int_{-\\infty}^{x}f(t)dt$.</li> '
                    '<li> The inverse of CDF $F(x)$, denoted by $F^{-1}(q)$, is called the '
                    '<b>Percent Point Function (PPF)</b>, where $q$ is the given cumulative '
                    'probability. This function is sometimes referred to as the <b>inverse '
                    'cumulative distribution function</b> or the <b>quantile function</b>.</li>'
                    '</ul>'),
                    'danger')
            
            with ui.card(height='620px'):
                ui.card_header('Standard normal distribution', style=chd_style)
                with ui.layout_sidebar():
                    with ui.sidebar(bg='#f8f8f8', width='350px'):  
                        ui.input_slider(id='crv_x', label='Value of the random variable:',
                                        min=-3.5, max=3.5, value=0.60, step=0.01)
                        
                        ui.markdown('---')

                        @render.ui
                        @reactive.event(input.crv_x)
                        def update_cvr_code():

                            x = input.crv_x()
                            pdf = norm.pdf(x)
                            cdf = norm.cdf(x)

                            return ui.markdown(('```python\n'
                                                f'norm.pdf({x}, loc=0, scale=1)\n'
                                                '```\n\n'
                                                f'<code>{pdf}</code>\n'
                                                '```python\n'
                                                f'norm.cdf({x}, loc=0, scale=1)\n'
                                                '```\n\n'
                                                f'<code>{cdf}</code>\n'
                                                '```python\n'
                                                f'norm.ppf({cdf.round(4)}, loc=0, scale=1)\n'
                                                '```\n\n'
                                                f'<code>{x}</code>\n'))
                    
                    ui.markdown(('The standard normal distribution is a normal distribution with '
                                'zero mean and unit standard deviation. Such a distribution is used '
                                'to illustrate the concepts of PDF, CDF, PPF, in the following '
                                'figure.'))

                    @render.plot(width=550, height=400)
                    @reactive.event(input.crv_x)
                    def update_norm_plot():
                        
                        step = 0.001
                        x = input.crv_x()
                        xs = np.arange(-3.5, 3.5+step, step)
                        pdf = norm.pdf(xs)
                        
                        fig, ax = plt.subplots()
                        ax.plot(xs, pdf, color='k', linewidth=2.5, alpha=0.6, label='PDF curve')
                        ax.fill_between(xs[xs<=x], y1=0, y2=pdf[xs<=x], color='b', alpha=0.4,
                                        label=f'CDF $P(X \\leq {x}) = {norm.cdf(x).round(4)}$')
                        ax.scatter(x, 0, s=40, c='r', alpha=0.5,
                                label=f'PPF $F^{{-1}}({norm.cdf(x).round(4)}) = {x}$')
                        ax.legend(fontsize=10, loc='upper right')
                        ax.grid()
                        ax.set_ylim([-0.03, 0.57])
                        ax.set_xlabel('Random variable $X$', fontsize=11)
                        ax.set_ylabel('PDF', fontsize=11)

                        return fig

        with ui.nav_panel("Random Sampling"):
            ui.markdown('### Distribution of Sample Means')
            ui.markdown(('Let $\\{X_1, X_2, ..., X_n\\}$ be a random sample of size $n$, *i.e.* a '
                        'a sequence of independent and identically distributed (i.i.d.) random '
                        'variables drawn from a population with an expected value $\\mu$ and finite '
                        'variance $\\sigma^2$, the sample mean is expressed as '
                        '$\\bar{x} = \\frac{1}{n}\\sum_{i=1}^nx_i$. \n'))
            ui_block(('<b>Central Limit Theorem:</b> For a relatively large sample size, the random '
                    'variable $\\bar{x} = \\frac{1}{n}\\sum_{i=1}^nx_i$ is approximately normally '
                    'distributed, regardless of the distribution of the population. The '
                    'approximation becomes better with increased sample size.'), 'danger')
            ui.markdown(('The following experiments are conducted to verify the Central Limit '
                        'Theorem (CLT). '))
            
            with ui.card(height='650px'):
                ui.card_header('Sampling Distribution of the Mean', style=chd_style)
                with ui.layout_sidebar():
                    with ui.sidebar(bg='#f8f8f8', width='350px'):  
                        ui.input_slider(id='sm_size', label='Sample size:',
                                        min=2, max=200, value=50, step=2)
                        dists = ['Uniform distribution between 0 and 1',
                                'Exponential distribution with unit mean',
                                'Standard normal distribution',
                                'Discrete uniform distribution for tossing a fair die',
                                'Bernoulli distribution with the probability to be 0.5']
                        funs = ['np.random.uniform(size=(1000, size))',
                                'np.random.exponential(size=(1000, size))',
                                'np.random.normal(size=(1000, size))',
                                'np.random.randint(1, 7, size=(1000, size))',
                                'np.random.binomial(1, 0.5, size=(1000, size))']
                        means = [0.5, 1, 0, 3.5, 0.5]
                        stds = [1/3**0.5/2, 1, 1, np.arange(1, 7).std(), 0.5]
                        ui.input_select(id='sm_dtype', label=ui.markdown('Distribution of $X_i$'),
                                        choices=dists, selected=dists[0])
                        ui.markdown('---')
                        
                        @render.data_frame
                        @reactive.event(input.sm_size, input.sm_dtype)
                        def update_sm_table():

                            np.random.seed(0)
                            size = input.sm_size()
                            dtype = input.sm_dtype()
                            idx = dists.index(dtype)
                        
                            samples = eval(funs[idx])
                            mean_exact = np.round(means[idx], 4)
                            std_exact = np.round(stds[idx] / size**0.5, 4)
                            mean_smp = np.round(samples.mean(), 4)
                            std_smp = np.round(samples.mean(axis=1).std(ddof=1), 4)

                            #return pd.DataFrame(np.random.rand(3, 5), columns=list('abcde'))

                            return pd.DataFrame({' ': ['Mean', 'STD'],
                                                'Exact': [mean_exact, std_exact],
                                                'Statistics': [mean_smp, std_smp]})

                    @render.plot(width=550, height=550)
                    @reactive.event(input.sm_size, input.sm_dtype)
                    def update_sm_plot():
                        
                        np.random.seed(0)
                        size = input.sm_size()
                        dtype = input.sm_dtype()
                        idx = dists.index(dtype)
                        
                        samples = eval(funs[idx])
                        mu = means[idx]
                        sigma = stds[idx]
                        xs = samples[0].copy()
                        xs.sort()
                        xm = samples.mean(axis=1)
                        xm.sort()
                        
                        fig, ax = plt.subplots(2, 2)
                        ax[0, 0].hist(xs, bins=25, color='b', alpha=0.5)
                        ax[0, 0].set_xlabel('Sample data value')
                        ax[0, 0].set_ylabel('Frequency')
                        ax[0, 0].set_title('Histogram of a sample', fontsize=10)
                        ax[0, 1].scatter(norm.ppf((np.arange(size)+0.5) / size),
                                        (xs - mu)/sigma, color='b', alpha=0.3)
                        ax[0, 1].plot([-3, 3], [-3, 3], linewidth=2, color='r', alpha=0.5)
                        ax[0, 1].set_xlabel('Standard normal quantiles')
                        ax[0, 1].set_ylabel('Ordered standard values')
                        ax[0, 1].set_title('Q-Q plot of the sample data', fontsize=10)
                        ax[1, 0].hist(xm, bins=25, color='b', alpha=0.5)
                        ax[1, 0].set_xlabel('Sample mean value')
                        ax[1, 0].set_ylabel('Frequency')
                        ax[1, 0].set_title('Histogram of 1000 sample means', fontsize=10)
                        ax[1, 1].scatter(norm.ppf((np.arange(1000)+0.5) / 1000),
                                        (xm - mu)/sigma*(size**0.5), color='b', alpha=0.3)
                        ax[1, 1].plot([-3, 3], [-3, 3], linewidth=2, color='r', alpha=0.5)
                        ax[1, 1].set_xlabel('Standard normal quantiles')
                        ax[1, 1].set_ylabel('Ordered standard values')
                        ax[1, 1].set_title('Q-Q plot of 1000 sample means', fontsize=10)

                        return fig

        with ui.nav_panel("Confidence Intervals"):
            ui.markdown('### Introduction to Confidence Intervals')
            ui.markdown(('Confidence interval provides a range of plausible values for the unknown '
                        'population parameter (such as the mean). The probability, or confidence '
                        'that the parameter lies in the confidence interval (i.e., that the '
                        'confidence interval contains the parameter), is called the confidence '
                        'level, denoted by $1-\\alpha$ in this lecture. If $1-\\alpha=95$%, for '
                        'instance, we are $95$% confident that the true population parameter lies '
                        'within the confidence interval. \n'))
        
            with ui.card(height='700px'):
                ui.card_header('Experiments on Confidence Intervals', style=chd_style)
                with ui.layout_sidebar():
                    with ui.sidebar(bg='#f8f8f8', width='350px'):  
                        ui.input_slider(id='ci_size', label='Sample size:',
                                        min=5, max=500, value=50, step=5)
                        ui.input_slider(id='ci_level', label='Confidence level',
                                        min=0.85, max=0.99, value=0.95, step=0.01)
                        std_info = ['Known standard deviation', 'Unknown standard deviation']
                        ui.input_select(id='ci_info', label='Population information',
                                        choices=std_info, selected=std_info[0])

                    ui.markdown(('In the following example, we repeat a sampling experiment 100 '
                                'times, and in each experiment, a sample with size $n$ is randomly '
                                'selected from a population uniformly distributed between 0 and 1. '
                                'The confidence interval for estimating the population mean is '
                                'calculated using the sample data and compared with the true '
                                'population parameter.'))
                    
                    @render.plot(width=550, height=290)
                    @reactive.event(input.ci_size, input.ci_level, input.ci_info)
                    def update_ci_plot():
                        
                        np.random.seed(0)
                        size = input.ci_size()
                        level = input.ci_level()
                        info = input.ci_info()

                        samples = np.random.rand(100, size)
                        if info == std_info[0]:
                            moe = norm.ppf(0.5 + level/2) * (1/2/3**0.5 / size**0.5) * np.ones(100)
                        else:
                            moe = norm.ppf(0.5 + level/2) * samples.std(axis=1, ddof=1) / size**0.5

                        fig, ax = plt.subplots()
                        idx = np.arange(1, 101)
                        means = samples.mean(axis=1)
                        ax.plot([-10, 110], [0.5, 0.5], color='b', linewidth=1.5, alpha=0.5)
                        is_out = (0.5 - moe > means) | (0.5 + moe < means)
                        ax.scatter(idx[~is_out], means[~is_out], s=10, c='k')
                        ax.scatter(idx[is_out], means[is_out], s=10, c='r')
                        ax.errorbar(idx[~is_out], means[~is_out],
                                    yerr=moe[~is_out], capsize=2, color='k', fmt='none')
                        ax.errorbar(idx[is_out], means[is_out],
                                    yerr=moe[is_out], capsize=2, color='r', fmt='none')
                        ax.set_xlabel('Experiments')
                        ax.set_xlim([-2, 102])
                        dev = (1/2/3**0.5 / size**0.5)
                        ax.set_ylim([0.5-dev*6, 0.5+dev*6])
                        ax.grid()

                        return fig
                
                    ui.markdown(('The definition of the confidence interval suggests that it covers '
                                'the true value of the population mean with a probability of '
                                '$1-\\alpha$. As a result, in the graph above, if $1-\\alpha=95$%,'
                                ' there are roughly (not always) $95$% of intervals (black lines) '
                                'capture the true population mean, while the remaining $5$% (red '
                                'lines) cases the population mean may fall out of the interval.'))

            ui.markdown('<br>')
            ui.markdown('### Calculations of Confidence Intervals')
            ui.markdown(('The equation for calculating the confidence interval when estimating the '
                        '**population mean** is presented below. The calculation is based on the '
                        'sample mean $\\bar{x}$ and the sample standard deviation $s$.'))
            ui_block(ci_equations('mean', 's'), 'danger')
            ui.markdown('The equation for calculating the confidence interval of the **population '
                        'proportion** is given as follows. ')
            ui_block(ci_equations('proportion', 'sigma'), 'danger')

            with ui.card(height='500px'):
                ui.card_header('Sample size for polling', style=chd_style)
                with ui.layout_sidebar():
                    with ui.sidebar(bg='#f8f8f8', width='350px'):  
                        ui.input_slider(id='polling_size', label='Sample size:',
                                        min=50, max=2500, value=1000, step=50)
                        ui.input_slider(id='polling_prop', label='Support rate of a candidate:',
                                        min=0.1, max=0.9, value=0.35, step=0.05)
                        ui.input_slider(id='polling_level', label='Confidence level:',
                                        min=0.85, max=0.99, value=0.95, step=0.01)
                
                    ui.markdown(('Political polling is usually used to predict the results of an '
                                'election. In this example, we focus on how 1) the sample size $n$; '
                                '2) the support rate $p$ of a candidate in the population; and 3) '
                                'the confidence level $1-\\alpha$, affect the credibility of a poll, '
                                'in terms of the margin of error.'))
                    
                    @render.plot(width=550, height=290)
                    @reactive.event(input.polling_size, input.polling_prop, input.polling_level)
                    def update_polling_plot():

                        size = input.polling_size()
                        p = input.polling_prop()
                        level = input.polling_level()

                        sizes = np.arange(40, 2501)
                        ci_curve = norm.ppf(0.5 + level/2) * ((p*(1-p))**0.5 / sizes**0.5)
                        ci_max = norm.ppf(0.5 + level/2) * (0.5 / sizes**0.5)
                        ci = norm.ppf(0.5 + level/2) * ((p*(1-p))**0.5 / size**0.5)

                        fig, ax = plt.subplots()
                        ax.plot(sizes, ci_curve*100, color='k', linewidth=1.5, alpha=0.55,
                                label=f'Margin of error with $p={p}$')
                        ax.plot(sizes, ci_max*100, 
                                color='m', linewidth=1.5, linestyle='--', alpha=0.55,
                                label=f'Maximum margin of error with $p=0.5$')
                        ax.scatter(size, ci*100, s=40, c='r', alpha=0.5)
                        ax.plot([0, size, size], [ci*100, ci*100, 0], 
                                color='r', linestyle='--', alpha=0.5)
                        ax.legend(fontsize=10)
                        ax.grid()
                        ax.set_xlabel('Sample size $n$', fontsize=11)
                        ax.set_ylabel('Margin of error (in percentage)', fontsize=11)
                        ax.set_xlim([0, 2580])
                        ax.set_ylim([0, 16.8])
                        
                        return fig

        with ui.nav_panel("Hypothesis Testing"):
            with ui.navset_card_underline(id="ht_estimate_type"):
                with ui.nav_panel("Population mean"):
                    ui.markdown(("**Step 1: Choose the hypotheses**<br>"
                                "The first step in setting up a hypothesis test is to decide on "
                                "the null hypothesis and the alternative hypothesis. <br><br>"))
                    with ui.navset_card_underline(id="ht_mean_test_type"):
                        with ui.nav_panel("Left-tail test"):
                            ui_block(('$$\n'
                                    '\\begin{cases}\n'
                                    'H_0:~\\mu \\geq \\mu_0 \\\\ \n'
                                    'H_a:~\\mu < \\mu_0 \n'
                                    '\\end{cases}\n'
                                    '$$\n'
                                    '<ul style="margin-bottom: 0px;">'
                                    '<li>$\\mu_0$ is the mean value assumed in the null '
                                    'hypothesis for testing.</li>'
                                    '</ul>'), 'danger')
                        with ui.nav_panel("Right-tail test"):
                            ui_block(('$$\n'
                                    '\\begin{cases}\n'
                                    'H_0:~\\mu \\leq \\mu_0 \\\\ \n'
                                    'H_a:~\\mu > \\mu_0 \n'
                                    '\\end{cases}\n'
                                    '$$\n'
                                    '<ul style="margin-bottom: 0px;">'
                                    '<li>$\\mu_0$ is the mean value assumed in the null '
                                    'hypothesis for testing.</li>'
                                    '</ul>'), 'danger')
                        with ui.nav_panel("Two-tail test"):
                            ui_block(('$$\n'
                                    '\\begin{cases}\n'
                                    'H_0:~\\mu = \\mu_0 \\\\ \n'
                                    'H_a:~\\mu \\not= \\mu_0 \n'
                                    '\\end{cases}\n'
                                    '$$\n'
                                    '<ul style="margin-bottom: 0px;">'
                                    '<li>$\\mu_0$ is the mean value assumed in the null '
                                    'hypothesis for testing.</li>'
                                    '</ul>'), 'danger')
                    ui.markdown("<br>")
                    ui.markdown("**Step 2: Compute the test statistic**<br><br>")
                    ui_block(('$$\n'
                            't_0 = \\frac{\\bar{x} - \\mu_0}{s/\\sqrt{n}}\n'
                            '$$\n'
                            '<ul style="margin-bottom: 0px;">'
                            '<li> Sample mean $\\bar{x} = \\frac{1}{n}\\sum_{i=1}^nx_i$.</li>\n'
                            '<li>Sample standard deviation $s$.</li>\n'
                            '<li>Sample size $n$.</li>'
                            '</ul>'), 'danger')
                    
                    ui.markdown("<br>")
                    ui.markdown(("**Step 3: Calculate the $P$-value**<br>"
                                "The $P$-value of a hypothesis test is the probability of getting "
                                "sample data at least as inconsistent with the null hypothesis "
                                "(and supportive of the alternative hypothesis) as the sample data "
                                "actually obtained. <br><br>"))

                    with ui.card(height='400px'):
                        ui.card_header('The $P$-value approach to hypothesis testing',
                                       style=chd_style)
                        with ui.layout_sidebar():
                            with ui.sidebar(bg='#f8f8f8', width='350px'):
                                ui.input_slider(id='t_test_value', label='Test statistic: value',
                                                min=-4, max=4, value=-1.5, step=0.001)
                                ui.input_slider(id='t_test_size', label='Sample size:',
                                                min=5, max=200, value=25, step=1)
                        
                            @render.plot(width=520, height=320)
                            @reactive.event(input.ht_mean_test_type, 
                                            input.t_test_value, input.t_test_size)
                            def update_t_test_type():
                            
                                test_type = input.ht_mean_test_type()
                                stat = input.t_test_value()
                                size = input.t_test_size()

                                xs = np.arange(-4, 4.001, 0.001)
                                pdf = t.pdf(xs, size-1)
                                
                                fig, ax = plt.subplots()
                                ax.plot(xs, pdf, color='k', linewidth=1.5, alpha=0.6,
                                        label='$t$-distribution PDF')
                                ax.set_title(f'{test_type} with $t_0={stat}$', fontsize=11)
                                ax.scatter(stat, t.pdf(stat, size-1), s=40, c='r', alpha=0.5)
                                ax.plot([stat, stat], [0, t.pdf(stat, size-1)], 
                                        color='r', linestyle='--',
                                        label=f'Test statistic $t_0={stat}$')

                                if test_type == 'Left-tail test':
                                    xs_left = xs[xs<=stat]
                                    pdf_left = t.pdf(xs_left, size-1)
                                    cdf_left = t.cdf(stat, size-1).round(4)
                                    ax.fill_between(xs_left, y1=0, y2=pdf_left,
                                                    color='b', alpha=0.4,
                                                    label=f'$P$-value = {cdf_left}')
                                elif test_type == 'Right-tail test':
                                    xs_right = xs[xs>=stat]
                                    pdf_right = t.pdf(xs_right, size-1)
                                    cdf_right = (1 - t.cdf(stat, size-1)).round(4)
                                    ax.fill_between(xs_right, y1=0, y2=pdf_right,
                                                    color='b', alpha=0.4,
                                                    label=f'$P$-value = {cdf_right}')
                                else:
                                    xs_both = xs.copy()
                                    xs_both[abs(xs) <= abs(stat)] = np.nan
                                    pdf_both = t.pdf(xs_both, size-1)
                                    cdf_both = (2 * (1 - t.cdf(np.abs(stat), size-1))).round(4)
                                    ax.fill_between(xs_both, y1=0, y2=pdf_both,
                                                    color='b', alpha=0.4,
                                                    label=f'$P$-value = {cdf_both}')

                                ax.set_xlabel('$t_0$ value', fontsize=11)
                                ax.set_ylabel('PDF', fontsize=11)
                                ax.legend(fontsize=10)
                                ax.grid()
                                ax.set_ylim([-0.03, 0.67])

                                return fig
                    
                    ui.markdown("<br>")
                    ui.markdown(("**Step 4: Make a decision**<br>"
                                "Given a significant level $\\alpha$, we draw conclusions from the "
                                "$P$-value. \n "))
                    ui_block(('<ul style="margin-bottom: 0px;">'
                              "<li> We reject the null hypothesis $H_0$ in favor of the alternative "
                              "hypothesis $H_a$, if the $P$-value is lower than the selected "
                              "significance level $\\alpha$.</li>"
                              "<li> Otherwise, we do not reject the null hypothesis.</li>"
                              "</ul>"), 'danger')

                with ui.nav_panel("Population proportion"):
                    ui.markdown(("**Step 1: Choose the hypotheses**<br>"
                                "The first step in setting up a hypothesis test is to decide on "
                                "the null hypothesis and the alternative hypothesis. <br><br>"))
                    with ui.navset_card_underline(id="ht_prop_test_type"):
                        with ui.nav_panel("Left-tail test"):
                            ui_block(('$$\n'
                                    '\\begin{cases}\n'
                                    'H_0:~p \\geq p_0 \\\\ \n'
                                    'H_a:~p < p_0 \n'
                                    '\\end{cases}\n'
                                    '$$\n'
                                    '<ul style="margin-bottom: 0px;">'
                                    '<li>$p_0$ is the proportion value assumed in the null '
                                    'hypothesis for testing.</li>'
                                    '</ul>'), 'danger')
                        with ui.nav_panel("Right-tail test"):
                            ui_block(('$$\n'
                                    '\\begin{cases}\n'
                                    'H_0:~p \\leq p_0 \\\\ \n'
                                    'H_a:~p > p_0 \n'
                                    '\\end{cases}\n'
                                    '$$\n'
                                    '<ul style="margin-bottom: 0px;">'
                                    '<li>$p_0$ is the proportion value assumed in the null '
                                    'hypothesis for testing.</li>'
                                    '</ul>'), 'danger')
                        with ui.nav_panel("Two-tail test"):
                            ui_block(('$$\n'
                                    '\\begin{cases}\n'
                                    'H_0:~p = p_0 \\\\ \n'
                                    'H_a:~p \\not= p_0 \n'
                                    '\\end{cases}\n'
                                    '$$\n'
                                    '<ul style="margin-bottom: 0px;">'
                                    '<li>$p_0$ is the proportion value assumed in the null '
                                    'hypothesis for testing.</li>'
                                    '</ul>'), 'danger')
                    ui.markdown("<br>")
                    ui.markdown("**Step 2: Compute the test statistic**<br><br>")
                    ui_block(('$$\n'
                            'z_0 = \\frac{\\hat{p} - p_0}{\\sqrt{p_0(1-p_0)/n}}\n'
                            '$$\n'
                            '<ul style="margin-bottom: 0px;">'
                            '<li> Sample proportion $\\hat{p}$.</li>\n'
                            '<li>Sample size $n$.</li>'
                            '</ul>'), 'danger')
                    
                    ui.markdown("<br>")
                    ui.markdown(("**Step 3: Calculate the $P$-value**<br>"
                                "The $P$-value of a hypothesis test is the probability of getting "
                                "sample data at least as inconsistent with the null hypothesis "
                                "(and supportive of the alternative hypothesis) as the sample data "
                                "actually obtained. <br><br>"))

                    with ui.card(height='400px'):
                        ui.card_header('The $P$-value approach to hypothesis testing',
                                    style=chd_style)
                        with ui.layout_sidebar():
                            with ui.sidebar(bg='#f8f8f8', width='350px'):
                                ui.input_slider(id='z_test_value', label='Test statistic: value',
                                                min=-4, max=4, value=-1.5, step=0.001)
                                ui.input_slider(id='z_test_size', label='Sample size:',
                                                min=5, max=200, value=25, step=1)
                        
                            @render.plot(width=520, height=320)
                            @reactive.event(input.ht_prop_test_type, 
                                            input.z_test_value, input.z_test_size)
                            def update_z_test_type():
                            
                                test_type = input.ht_prop_test_type()
                                stat = input.z_test_value()

                                xs = np.arange(-4, 4.001, 0.001)
                                pdf = norm.pdf(xs)
                                
                                fig, ax = plt.subplots()
                                ax.plot(xs, pdf, color='k', linewidth=1.5, alpha=0.6,
                                        label='Standard normal distribution PDF')
                                ax.set_title(f'{test_type} with $t_0={stat}$', fontsize=11)
                                ax.scatter(stat, norm.pdf(stat), s=40, c='r', alpha=0.5)
                                ax.plot([stat, stat], [0, norm.pdf(stat)], color='r', linestyle='--',
                                        label=f'Test statistic $z_0={stat}$')

                                if test_type == 'Left-tail test':
                                    xs_left = xs[xs<=stat]
                                    pdf_left = norm.pdf(xs_left)
                                    cdf_left = norm.cdf(stat).round(4)
                                    ax.fill_between(xs_left, y1=0, y2=pdf_left, color='b', alpha=0.4,
                                                    label=f'$P$-value = {cdf_left}')
                                elif test_type == 'Right-tail test':
                                    xs_right = xs[xs>=stat]
                                    pdf_right = norm.pdf(xs_right)
                                    cdf_right = (1 - norm.cdf(stat)).round(4)
                                    ax.fill_between(xs_right, y1=0, y2=pdf_right, color='b', alpha=0.4,
                                                    label=f'$P$-value = {cdf_right}')
                                else:
                                    xs_both = xs.copy()
                                    xs_both[abs(xs) <= abs(stat)] = np.nan
                                    pdf_both = norm.pdf(xs_both)
                                    cdf_both = (2 * (1 - norm.cdf(np.abs(stat)))).round(4)
                                    ax.fill_between(xs_both, y1=0, y2=pdf_both, color='b', alpha=0.4,
                                                    label=f'$P$-value = {cdf_both}')

                                ax.set_xlabel('$z_0$ value', fontsize=11)
                                ax.set_ylabel('PDF', fontsize=11)
                                ax.legend(fontsize=10)
                                ax.grid()
                                ax.set_ylim([-0.03, 0.67])

                                return fig
                    
                    ui.markdown("<br>")
                    ui.markdown(("**Step 4: Make a decision**<br>"
                                "Given a significant level $\\alpha$, we draw conclusions from the "
                                "$P$-value. \n "))
                    ui_block(('<ul style="margin-bottom: 0px;">'
                              "<li> We reject the null hypothesis $H_0$ in favor of the alternative "
                              "hypothesis $H_a$, if the $P$-value is lower than the selected "
                              "significance level $\\alpha$.</li>"
                              "<li> Otherwise, we do not reject the null hypothesis.</li>"
                              "</ul>"), 'danger')
        
        with ui.nav_panel("Interpretation of Linear Regression"):
            ui.markdown('### Notations:')
            ui.markdown('A linear regression model can be written as'
                        '$$\n'
                        'y = \\beta_0 + \\beta_1x_1 + \\beta_2x_2 + \\cdots + \\beta_px_p + u\n'
                        '$$\n'
                        'where $y$ is the dependent variable, $x_1$, $x_2$, ..., $x_p$ are ' 
                        'independent variables, the term $u$ represents the random error, and '
                        '$\\beta_0$, $\\beta_1$, $\\beta_2$, ..., $\\beta_p$ are model parameters.')
            with ui.card(height='650px'):
                ui.card_header('Illustration of notations', style=chd_style)
                with ui.layout_sidebar():
                    with ui.sidebar(bg='#f8f8f8', width='350px'):  
                        notations = ['Population regression function',
                                     'Sample regression function',
                                     'Residuals']
                        ui.input_selectize(id='display_reg', label='Notations',
                                           choices=notations, selected=notations[0], multiple=True)
                        ui.input_action_button(id='data_gen', label='Generate new dataset')
                    
                    ui.markdown('A simple linear regression model $y = \\beta_0 + \\beta_1 + u$ is '
                                'used to illustrate the notations. Here, the population parameters '
                                '$\\beta_0=1.0$ and $\\beta_1=5.0$, and the random error $u$ follows '
                                'a standard normal distribution $N(0, 1)$.')
                    
                    ns = 25
                    xd = np.random.rand(ns)
                    yd = 1 + 5*xd + np.random.normal(size=ns)
                    samples = reactive.value((xd, yd))

                    @reactive.effect
                    @reactive.event(input.data_gen)
                    def generate_data():
                        
                        xd = np.random.rand(ns)
                        yd = 1 + 5*xd + np.random.normal(size=ns)
                        samples.set((xd, yd))

                    @render.plot(width=550, height=480)
                    @reactive.event(input.display_reg, input.data_gen)
                    def update_reg_plot():
                        
                        xd, yd = samples.get()
                        res = linregress(xd, yd)
                        beta0, beta1 = res.intercept, res.slope
                        
                        displays = input.display_reg()

                        fig, ax = plt.subplots()
                        ax.scatter(xd, yd, color='r', alpha=0.6, label='The sample data')
                        if notations[0] in displays:
                            ax.plot([0, 1], [1, 6], color='m', linewidth=1.5, linestyle='--',
                                    label='PRF $\\mathbb{E}(y|x_1) = \\beta_0 + \\beta_1 x_1$')
                        if notations[1] in displays:
                            ax.plot([0, 1], [beta0, beta0+beta1], color='b', linewidth=1.5,
                                    label='SRF $\\hat{y} = \\hat{\\beta}_0 + \\hat{\\beta}_1 x_1$')
                            ax.text(0.65, 0.55, f'$\\hat{{\\beta}}_0={beta0:.4f}$\n', fontsize=12)
                            ax.text(0.65, -0.25, f'$\\hat{{\\beta}}_1={beta1:.4f}$\n', fontsize=12)
                        if notations[2] in displays:
                            ax.vlines(xd, yd, beta0+beta1*xd, color='k', linewidth=1,
                                    label='Residuals')
                        ax.legend(fontsize=10, loc='upper left')
                        ax.set_xlabel('Independent variable $x_1$', fontsize=11)
                        ax.set_ylabel('Dependent variable $y$', fontsize=11)
                        ax.grid()
                        ax.set_ylim([-1.7, 8.7])
                        ax.set_xlim([-0.03, 1.03])

                        return fig
            
            ui.markdown('<br>')
            ui.markdown('### Goodness-of-Fit')
            ui.markdown(('The $R^2$-value, sometimes called the **coefficient of determination** of '
                        'a regression model, defined below, is a measure of how well the model fits '
                        'the observed data.\n'
                        '$$\n'
                        'R^2 = \\frac{\\text{SSE}}{\\text{SST}} = 1-\\frac{\\text{SSR}}{\\text{SST}}\n'
                        '$$\n'
                        'where\n'
                        '- SST: the **total sum of squares**, which measures the total variations '
                        'of the sample data $y_i$.\n'
                        '- SSE: the **explained sum of squares**, which measures the variation in '
                        'the fitted value $\\hat{y}_i$.\n'
                        '- SSR: the **residual sum of squares**, which measures the variation in '
                        'the residuals $\\hat{u}_i$.'))

            with ui.card(height='650px'):
                ui.card_header('Illustration of the $R^2$ value', style=chd_style)
                with ui.layout_sidebar():
                    with ui.sidebar(bg='#f8f8f8', width='350px'):
                        choices = ['SST', 'SSE', 'SSR']
                        ui.input_radio_buttons(id='r_square_comp', label='Components of variations:',
                                            choices=choices, selected=choices[0])
                        ui.input_slider(id='error_scale', label='Scale of the error term:',
                                        min=0.5, max=2.5, value=1.0, step=0.01)

                    ui.markdown('The data visual below is used to illustrate that the variation '
                                'components follow the equation $\\text{SST} = \\text{SSE} + '
                                '\\text{SSR}$.')
                    
                    @render.plot(width=550, height=480)
                    @reactive.event(input.r_square_comp, input.error_scale, input.data_gen)
                    def update_rsquare_plot():
                        
                        xd, yd = samples.get()
                        escale = input.error_scale()
                        err = (yd - 1 - 5*xd)
                        yd = 1 + 5*xd +  err*escale
                        res = linregress(xd, yd)
                        beta0, beta1, rsquare = res.intercept, res.slope, res.rvalue**2
                        
                        fig, ax = plt.subplots()
                        ax.scatter(xd, yd, color='r', alpha=0.6, label='The sample data')
                        ax.plot([0, 1], [beta0, beta0+beta1], color='b', linewidth=1.5,
                                label='SRF $\\hat{y} = \\hat{\\beta}_0 + \\hat{\\beta}_1 x_1$')
                        ybar = yd.mean()
                        ax.plot([0, 1], np.array([1, 1])*ybar,
                                color='m', linewidth=1.5, linestyle='--',
                                label='Sample average $\\bar{y}$')
                        if input.r_square_comp() == 'SST':
                            ax.vlines(xd, yd, ybar, color='r', label='SST')
                        if input.r_square_comp() == 'SSE':
                            ax.vlines(xd, beta0+beta1*xd, ybar, color='g',label='SSE')
                        if input.r_square_comp() == 'SSR':
                            ax.vlines(xd, yd, beta0+beta1*xd, color='k', label='SSR')
                        ax.legend(fontsize=10, loc='upper left')
                        ax.text(0.62, -1.8, f'$R^2\\text{{ value}}={rsquare:.4f}$\n', fontsize=12)
                        sst = ((yd - yd.mean())**2).sum()
                        ax.text(0.62, -3.2, f'$\\text{{SST}}={sst:.4f}$\n', fontsize=12)
                        sse = ((beta0 + beta1*xd - yd.mean())**2).sum()
                        ax.text(0.62, -4.6, f'$\\text{{SSE}}={sse:.4f}$\n', fontsize=12)
                        ssr = ((yd - beta0 - beta1*xd)**2).sum()
                        ax.text(0.62, -6, f'$\\text{{SSR}}={ssr:.4f}$\n', fontsize=12)
                        ax.set_xlabel('Independent variable $x_1$', fontsize=11)
                        ax.set_ylabel('Dependent variable $y$', fontsize=11)
                        ax.grid()
                        ax.set_ylim([-5.8, 10.8])
                        ax.set_xlim([-0.03, 1.03])

                        return fig

        with ui.nav_panel("Predictive Modeling: Regression"):
            ui.markdown('### Bias-Variance Tradeoff')
            ui.markdown("The **bias–variance tradeoff** describes the relationship between a model's "
                        "complexity, the accuracy of its predictions, and how well it can make "
                        "predictions on previously unseen data that were not used to train the model. "
                        "In general, as we increase the number of tunable parameters in a model, it "
                        "becomes more flexible, and can better fit a training data set. It is said "
                        "to have lower error, or bias. However, for more flexible models, there will "
                        "tend to be greater variance to the model fit each time we take a set of "
                        "samples to create a new training data set. As a result, the best model "
                        "performance may be achieved by miniminzing the combination of bias and "
                        "variance.")

            with ui.card(height='850px'):
                ui.card_header('Bias-variance tradeoff in regression models', style=chd_style)
                with ui.layout_sidebar():
                    with ui.sidebar(bg='#f8f8f8', width='350px'):
                        with ui.layout_columns(col_widths=(8, 4)):
                            ui.HTML(f'<p style="padding-top:5pt">Samples of training data:</p>')
                            ui.input_selectize("reg_data_size", label="",
                                               choices=[30, 50, 100, 200], selected=50)
                        ui.input_action_button('polyreg_gen', 'Repeat the experiment')
                        
                        displays = ['Population regression function', 'Training data', 'Test data']
                        ui.input_selectize(id='polyreg_displays', label='Display options',
                                           choices=displays, selected=displays[1], multiple=True)
                        
                        model_choices = ["None", "Polynomial regression", "K-nearest neighbors",
                                         "Decision tree", "Bagged trees"]
                        ui.input_selectize("reg_model", "Regression model:", choices=model_choices)
                        
                        @render.express
                        def reg_model_params_ui():
                            model = input.reg_model()
                            if model == "Polynomial regression":
                                ui.input_slider('polyreg_k', 'The number of polynomial terms',
                                                min=1, max=25, value=4, step=1)
                            elif model == "K-nearest neighbors":
                                ui.input_slider('knn_reg_k', 'The number of neighbors',
                                                min=1, max=25, value=5, step=1)
                            elif model in ["Decision tree", "Bagged trees"]:
                                #with ui.layout_columns(col_widths=(6, 6)):
                                with ui.layout_columns(col_widths=(6, 6), gap="10px"):
                                    ui.HTML(f'<p style="padding-top:16pt">Max leaves:</p>')
                                    ui.input_slider('dtree_reg_max_leaf', "",
                                                     min=2, max=40, value=8, step=1)
                                    
                                    ui.HTML(f'<p style="padding-top:16pt">Max tree depth:</p>')
                                    ui.input_slider('dtree_reg_depth', "",
                                                     min=1, max=10, value=4, step=1)

                                    ui.HTML(f'<p style="padding-top:16pt">Min leaf samples:</p>')
                                    ui.input_slider('dtree_reg_mins_sample_leaf', '',
                                                     min=1, max=30, value=1, step=1)

                                    ui.HTML(f'<p style="padding-top:16pt">Min split samples:</p>')
                                    ui.input_slider('dtree_reg_mins_sample_split', '',
                                                     min=2, max=50, value=2, step=1)

                    ui.markdown("A training dataset is generated from a mysterious function, and "
                                "you may use a regression model below with specific hyper-parameters "
                                "to predict the value of the dependent variable $y$. "
                                "Repeat the experiments to explore the tradeoff between the bias "
                                "and variance of the prediction.")
                    
                    train_size = 50
                    total_size = train_size + 50
                    this_data = mysterious(total_size)
                    all_samples = reactive.value([this_data])
                    
                    @reactive.effect
                    @reactive.event(input.polyreg_gen)
                    def generate_polyreg_data():

                        train_size = input.reg_data_size()
                        if train_size != "":
                            content = all_samples.get()
                            content.append(mysterious(int(train_size) + 25))

                    @reactive.effect
                    @reactive.event(input.reg_data_size)
                    def reset_polyreg_data():
                        
                        train_size = input.reg_data_size()
                        if train_size != "":
                            all_samples.set([mysterious(int(train_size) + 25)])
                    
                    @render.plot(width=550, height=600)
                    def update_polyreg_plot():

                        input.polyreg_gen()
                        samples = all_samples.get()    
                        current_sample = samples[-1]

                        train_size = input.reg_data_size()
                        train_size = int(train_size) if train_size != "" else 50
                        train = current_sample.loc[:train_size]
                        test = current_sample.loc[train_size:]

                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6), height_ratios=[2, 1])
                        xs = np.arange(0, 1.01, 0.01)
                        if "Population regression function" in input.polyreg_displays():
                            ax1.plot(xs, mysterious_prf(xs),
                                     color='g', linewidth=2, label='PRF')
                        if "Training data" in input.polyreg_displays():
                            ax1.scatter(train['x'], train['y'],
                                        color='b', alpha=0.4, label='Training data')
                        if "Test data" in input.polyreg_displays():
                            ax1.scatter(test['x'], test['y'],
                                        color='r', alpha=0.4, label='Test data')
                        
                        params = {}
                        model = input.reg_model()
                        if model == "Polynomial regression":
                            params["degree"] = input.polyreg_k()
                        elif model == "K-nearest neighbors":
                            params["n_neighbors"] = input.knn_reg_k()
                        elif model in ["Decision tree", "Bagged trees"]:
                            params["max_leaf_nodes"] = input.dtree_reg_max_leaf()
                            params["max_depth"] = input.dtree_reg_depth()
                            params["min_samples_leaf"] = input.dtree_reg_mins_sample_leaf()
                            params["min_samples_split"] = input.dtree_reg_mins_sample_split()

                        rmse_dict = dict(train=[], test=[])
                        for i, each_sample in enumerate(samples):
                            each_train = each_sample.loc[:train_size]
                            each_test = each_sample.loc[train_size:]
                            res = reg_pred(each_train, model, params)
                            if res is not None:
                                x_pred, y_pred, pipe = res
                                if i < len(samples) - 1:
                                    ax1.plot(x_pred, y_pred, color='k', linewidth=1,
                                             linestyle='--', alpha=0.2)
                                else:
                                    ax1.plot(x_pred, y_pred, color='m', linewidth=1.5,
                                             label='Prediction')
                                
                                rmse_train = rmse(each_train['y'], pipe.predict(each_train[['x']]))
                                rmse_test = rmse(each_test['y'], pipe.predict(each_test[['x']]))
                                rmse_dict["train"].append(rmse_train)
                                rmse_dict["test"].append(rmse_test)

                        ax1.set_xlabel('Predictor variable $x$', fontsize=11)
                        ax1.set_ylabel('Predicted variable $y$', fontsize=11)
                        ax1.set_ylim([-3.8, 27.3])
                        ax1.grid()
                        if len(input.polyreg_displays()) > 0:
                            ax1.legend(fontsize=10, loc='upper center')
                        
                        #ax2.boxplot([rmse_dict["train"], rmse_dict["test"]], labels=['Train', 'Test'])
                        if len(samples) == len(rmse_dict["train"]):
                            ax2.plot(range(len(samples)), rmse_dict["train"],
                                     color='b', marker='o', alpha=0.4, label="Training")
                            ax2.plot(range(len(samples)), rmse_dict["test"],
                                     color='r', marker='o', alpha=0.4, label="Test")
                            ax2.hlines(np.mean(rmse_dict["train"]), xmin=-0.4, xmax=len(samples)-0.5,
                                       color='b', linestyle='--')
                            ax2.hlines(np.mean(rmse_dict["test"]), xmin=-0.4, xmax=len(samples)-0.5,
                                       color='r', linestyle='--')
                            ax2.legend(fontsize=10, loc="upper left")
                            ax2.set_xlim([-0.4, len(samples)-0.6])
                            rmse_max = max([max(rmse_dict["train"]), max(rmse_dict["test"])])
                            if rmse_max > 200:
                                ymax = max([np.mean(rmse_dict["test"]), 200])
                                ax2.set_ylim([-6, ymax*1.2])
                            #ax2.set_ylim([1, 10 ** np.ceil(np.log10(rmse_max))])
                            #ax2.set_ylim([-rmse_max*0.1,
                            #              max([rmse_max*1.25, np.mean(rmse_dict['test'])*1.85])])
                            
                            
                        ax2.set_xlabel("Experiements")
                        ax2.set_ylabel("RMSE", fontsize=11)
                        #ax2.set_yscale('symlog')
                        ax2.grid()

                        return fig
                
                #@render.plot(width=550, height=200)
                #def update_polyreg_records_plot():

                #    ax, fig = plt.subplots(figsize=(5.5, 2))

        with ui.nav_panel("Predictive Modeling: Classification"):
            ui.markdown('### Fitting classes of various patterns')
            ui.markdown("Choosing a suitable classification model depends heavily on "
                        "the characteristics of the data and the specific problem. For "
                        "example, logistic regression models and support vector machines "
                        "with linear kernels are effective in dealing with linearly "
                        "separable classes, while tree-based methods are more suitable "
                        "for non-linearly separable data. Besides, for a selected model, "
                        "its hyper-parameters will affect the **bias-variance tradeoff**, "
                        "thus influencing model performance.")
            with ui.card(height='850px'):
                ui.card_header('Illustration of different classification models', style=chd_style)
                with ui.layout_sidebar():
                    with ui.sidebar(bg='#f8f8f8', width='350px'):
                        with ui.layout_columns(col_widths=(8, 4)):
                            ui.HTML(f'<p style="padding-top:5pt">Sample size of each class:</p>')
                            ui.input_selectize("class_data_size", label="",
                                               choices=[15, 25, 50, 100], selected=25)
                        ui.markdown('Separable by a hyperplane:')
                        separable_choices = ['Separable', 'Barely separable', 'Non-separable']
                        ui.input_radio_buttons('class_pattern', label='',
                                               choices=separable_choices)
                        ui.input_action_button('classes_gen', label='Generate new dataset')
                        
                        model_choices = ['None', 'Logistic regression',
                                         'Linear discriminant analysis',
                                         'Quadratic discriminant analysis',
                                         'Support vector machine',
                                         'K-nearest neighbors',
                                         'Decision tree', 'Bagged trees']
                        ui.input_selectize('class_model', label='Classification model:',
                                           choices=model_choices,)

                        with ui.navset_hidden(id='model_display_options'):
                            with ui.nav_panel("model_off"):
                                None
                            with ui.nav_panel("model_on"):
                                ui.input_selectize("class_display", label="Display options",
                                                   choices=["Prediction", "Probabilities"],
                                                   selected=["Prediction"],
                                                   multiple=True, remove_button=True)
                                
                                @render.express
                                def cls_model_params_ui():
                                    model = input.class_model()
                                    if model == "Logistic regression":
                                        ui.input_slider('logreg_alpha', 'Regularization parameter alpha:',
                                                        min=0.0, max=30, value=0, step=0.01)
                                    elif model == "Support vector machine":
                                        ui.input_selectize('svm_kernel_selectize', 'Kernel type:',
                                                           choices=['linear', 'poly', 'rbf'],
                                                           selected='linear')
                                        ui.input_slider('svm_alpha', 'Regularization parameter alpha:',
                                                        min=0.01, max=10, value=1, step=0.01)
                                                                                    
                                        with ui.navset_hidden(id='svm_degree_ui'):
                                            with ui.nav_panel('svm_poly_panel'):
                                                ui.input_slider('svm_poly_degree', 'Polynomial degree:',
                                                                min=1, max=10, value=3, step=1)
                                            with ui.nav_panel('svm_not_poly_panel'):
                                                None
                                        
                                        @reactive.effect
                                        @reactive.event(input.svm_kernel_selectize)
                                        def toggle_svm_param_options():
                                            if input.svm_kernel_selectize() == 'poly':
                                                ui.update_navset('svm_degree_ui', selected='svm_poly_panel')
                                            else:
                                                ui.update_navset('svm_degree_ui', selected='svm_not_poly_panel')

                                    elif model == "K-nearest neighbors":
                                        ui.input_slider('knn_cls_k', 'Number of neighbors K:',
                                                        min=1, max=30, value=5, step=1)
                                    elif model in ["Decision tree", "Bagged trees"]:
                                        with ui.layout_columns(col_widths=(6, 6), gap="10px"):
                                            ui.HTML(f'<p style="padding-top:16pt">Max leaves:</p>')
                                            ui.input_slider('dtree_cls_max_leaf', "",
                                                             min=2, max=40, value=8, step=1)
                                            ui.HTML(f'<p style="padding-top:16pt">Max tree depth:</p>')
                                            ui.input_slider('dtree_cls_depth', "",
                                                             min=1, max=10, value=4, step=1)
                                            ui.HTML(f'<p style="padding-top:16pt">Min leaf samples:</p>')
                                            ui.input_slider('dtree_cls_mins_sample_leaf', '',
                                                             min=1, max=30, value=1, step=1)
                                            ui.HTML(f'<p style="padding-top:16pt">Min split samples:</p>')
                                            ui.input_slider('dtree_cls_mins_sample_split', '',
                                                             min=2, max=50, value=2, step=1)

                                @reactive.effect
                                @reactive.event(input.class_model)
                                def update_class_display_choices():
                                    if input.class_model() in ["Linear discriminant analysis",
                                                               "Quadratic discriminant analysis"]:
                                        display_choices = ["Prediction", "Probabilities", "PDF countours"]
                                    elif input.class_model() == "Support vector machine":
                                        display_choices = ["Prediction", "Margin", "Support vectors"]
                                    else:
                                        display_choices = ["Prediction", "Probabilities"]
                                    ui.update_selectize("class_display", choices=display_choices,
                                                        selected=["Prediction"])
                        
                        @reactive.effect
                        @reactive.event(input.class_model)
                        def toggle_model_options():
                            if input.class_model() == 'None':
                                ui.update_navset("model_display_options", selected="model_off")
                            else:
                                ui.update_navset("model_display_options", selected="model_on")
                
                    
                    size = 50
                    class_samples = reactive.value(two_class_data(size))

                    @reactive.effect
                    @reactive.event(input.classes_gen, input.class_data_size,
                                    ignore_init=True)
                    def generate_all_class_data():
                        size = input.class_data_size()
                        if size != "":
                            class_samples.set(two_class_data(int(size)))

                    
                    ui.markdown(
                        "Here, we create a simulated dataset where there are two predictor "
                        "variables, $x_1$ and $x_2$, and the response variable $y$ may belong "
                        "to one of two classes. These two classes may be 1) separable by a "
                        "hyperplane; 2) barely separable by a hyperplane; or 3) non-separable "
                        "by a hyperplane."
                    )

                    @render.plot(height=550, width=550)
                    def update_class_plot():
                        colors = ['r', 'b']

                        pattern = input.class_pattern()
                        if pattern == "":
                            return
                        samples = class_samples.get()[pattern]

                        model = input.class_model()
                        params = {}
                        if model == "Logistic regression":
                            params['max_iter'] = 1000000
                            alpha_value = input.logreg_alpha()
                            if alpha_value is None or alpha_value == 0:
                                params['penalty'] = None
                            else:
                                params['C'] = 1 / alpha_value
                        elif model == "Support vector machine":
                            params['kernel'] = input.svm_kernel_selectize()
                            alpha_value = input.svm_alpha()
                            if alpha_value is None or alpha_value == 0:
                                params['C'] = 1e5
                            else:
                                params['C'] = 1 / alpha_value
                            if params['kernel'] == 'poly':
                                params['degree'] = input.svm_poly_degree()
                                params['coef0'] = 1
                        elif model == 'K-nearest neighbors':
                            params['n_neighbors'] = input.knn_cls_k()
                        elif model in ['Decision tree', 'Bagged trees']:
                            params['max_leaf_nodes'] = input.dtree_cls_max_leaf()
                            params['max_depth'] = input.dtree_cls_depth()
                            params['min_samples_leaf'] = input.dtree_cls_mins_sample_leaf()
                            params['min_samples_split'] = input.dtree_cls_mins_sample_split()
                        res = two_class_pred(samples, model, params)

                        samples_renamed = samples.copy()
                        if res is not None:
                            y_pred = res[-1].predict(samples[['x1', 'x2']])
                            is_correct = (y_pred == samples['y'])
                            ca1, ca2 = (is_correct[samples['y']=='Class 1'].mean(),
                                        is_correct[samples['y']=='Class 2'].mean())
                            c1_label, c2_label = f"Class 1 accuray: {ca1:.3f}", f"Class 2 accuray: {ca2:.3f}"
                            samples_renamed["y"] = samples['y'].map({'Class 1': c1_label, 'Class 2': c2_label})

                        max_den = 0.47 if pattern == 'Non-separable' else 0.27
                        data_kws = dict(edgecolor="black", alpha=0.4, linewidth=1)
                        plots = sns.jointplot(data=samples_renamed, x="x1", y="x2",
                                              xlim=(-3.2, 5.8), ylim=(-3.2, 5.8),
                                              hue="y", palette=colors,
                                              marginal_ticks=True, ratio=4,
                                              joint_kws=data_kws, marginal_kws=data_kws)
                        sns.despine(top=False, right=False)
                        plots.ax_joint.set_xlabel('Predictor variable $x_1$', fontsize=11)
                        plots.ax_joint.set_ylabel('Predictor variable $x_2$', fontsize=11)
                        plots.ax_joint.legend(fontsize=10, loc='upper left',
                                              facecolor='white', edgecolor='black')
                        plots.ax_marg_x.grid()
                        plots.ax_marg_x.set_ylim(0, max_den)
                        plots.ax_marg_y.grid()
                        plots.ax_marg_y.set_xlim(0, max_den)
                        plots.ax_joint.grid()

                        if res is not None:
                            xx1, xx2, proba_pred, pipe = res
                            if "Prediction" in input.class_display():
                                plots.ax_joint.contourf(xx1, xx2, 1 - proba_pred,
                                                        levels = [-0.5, 0.5, 1.5],
                                                        alpha=0.15, colors=colors)
                                plots.ax_joint.contour(xx1, xx2, proba_pred, levels=[0.49999]) 
                            if "Probabilities" in input.class_display():
                                cs = plots.ax_joint.contour(xx1, xx2, proba_pred, 
                                                            levels=[0.1, 0.3, 0.7, 0.9],
                                                            colors='k', linestyles='dashed')
                                plots.ax_joint.clabel(cs, fmt='%1.1f', fontsize=10)
                            if "PDF countours" in input.class_display():
                                c1 = samples.loc[samples["y"]=="Class 1", ["x1", "x2"]].values
                                c2 = samples.loc[samples["y"]=="Class 2", ["x1", "x2"]].values
                                mu1 = c1.mean(axis=0)
                                mu2 = c2.mean(axis=0)
                                if model == "Linear discriminant analysis":
                                    devs = np.vstack((c1 - mu1, c2 - mu2))
                                    cov1 = cov2 = np.cov(devs.T, ddof=1)
                                else:
                                    cov1 = np.cov(c1.T, ddof=1)
                                    cov2 = np.cov(c2.T, ddof=1)
                                xx = np.array([xx1, xx2]).transpose((1, 2, 0))
                                y1 = multivariate_normal.pdf(xx, mu1, cov1)
                                y2 = multivariate_normal.pdf(xx, mu2, cov2)
                                yc = np.maximum(y1, y2)
                                levels = np.percentile(yc.reshape(yc.size),
                                                       [15, 40, 60, 80, 90, 95, 99])
                                plots.ax_joint.contour(xx1, xx2, yc, levels=levels, cmap='plasma',
                                                       linestyles="solid", linewidths=2, alpha=0.6)
                            if "Margin" in input.class_display():
                                xx = pd.DataFrame({'x1': xx1.ravel(), 'x2': xx2.ravel()})
                                yd = pipe.decision_function(xx)
                                yd = yd.reshape(xx1.shape)
                                plots.ax_joint.contour(xx1, xx2, yd, levels=[-1, 1],
                                                       colors='k', linestyles='dashed')
                            if "Support vectors" in input.class_display():
                                svec = pipe.named_steps['cls'].support_vectors_
                                plots.ax_joint.scatter(svec[:, 0], svec[:, 1], s=150,
                                                       marker='o', color='none', 
                                                       linewidth=1.5, edgecolor='k')
                        
                        return plots.figure

                         
        with ui.nav_panel("About"):
            ui.markdown('Under development.')

