from django.shortcuts import render
import numpy as np
from scipy import stats
import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import re

def descriptive(request):
    tab = request.GET.get('tab', 'bulk')

    context = {
        'segment': 'descriptive',
        'active_tab': tab,
        'data': request.session.get('data', ""),
        'results': request.session.get('results', None),
        'graph': request.session.get('graph', None),
        'cv_graph': request.session.get('cv_graph', None),
        'normality_graph': request.session.get(
            'normality_graph',
            None
        ),
        'shape_graph': request.session.get(
            'shape_graph',
            None
        ),
        'graph_h': request.session.get('graph_h', None),
        'histogram_variables': request.session.get(
            'histogram_variables',
            []
        ),

        'histogram_selected_variable': request.session.get(
            'histogram_selected_variable',
            None
        ),

        'histogram_bin_method': request.session.get(
            'histogram_bin_method',
            'auto'
        ),

        'histogram_custom_bins': request.session.get(
            'histogram_custom_bins',
            ''
        ),

        'histogram_selected_bins': request.session.get(
            'histogram_selected_bins',
            None
        ),

        'histogram_method_label': request.session.get(
            'histogram_method_label',
            'Automatic'
        ),
        'boxplot': request.session.get('boxplot', None),
        'headers': request.session.get('headers', None),
        'use_first_row_as_header': 'checked' if request.session.get('use_first_row_as_header', False) else '',
    }

    if request.method == "POST" and request.POST.get("clear") == "true":
        if 'data' in request.session:
            del request.session['data']
        if 'results' in request.session:
            del request.session['results']
        if 'graph' in request.session:
            del request.session['graph']
        if 'cv_graph' in request.session:
            del request.session['cv_graph']
        if 'normality_graph' in request.session:
            del request.session['normality_graph']
        if 'shape_graph' in request.session:
            del request.session['shape_graph']
        if 'graph_h' in request.session:
            del request.session['graph_h']
        if 'boxplot' in request.session:
            del request.session['boxplot']
        if 'headers' in request.session:
            del request.session['headers']
        if 'use_first_row_as_header' in request.session:
            request.session.pop('use_first_row_as_header', False)

        for key in [
            'histogram_settings',
            'histogram_variables',
            'histogram_selected_variable',
            'histogram_bin_method',
            'histogram_custom_bins',
            'histogram_selected_bins',
            'histogram_method_label',
        ]:
            request.session.pop(key, None)

        context['data'] = ""
        context['results'] = None
        context['graph'] = None
        context['cv_graph'] = None
        context['normality_graph'] = None
        context['shape_graph'] = None
        context['graph_h'] = None
        context['boxplot'] = None
        context['use_first_row_as_header'] = False
        context['histogram_variables'] = []
        context['histogram_selected_variable'] = None
        context['histogram_bin_method'] = 'auto'
        context['histogram_custom_bins'] = ''
        context['histogram_selected_bins'] = None
        context['histogram_method_label'] = 'Automatic'

        return render(request, "descriptive/descriptive.html", context)

####################################################################################################################
    if request.method == "POST" and tab == "bulk":
        data = request.POST.get('data')
        use_first_row_as_header = request.POST.get('use_first_row_as_header') == 'on'

        if not data.strip():
            context['error'] = "Please enter data before calculating."
            context['results'] = None
            context['graph'] = None
            return render(request, "descriptive/descriptive.html", context)

        data = data.replace('\r', '').strip()
        rows = [row.split('\t') for row in data.split('\n')]
        columns = []
        headers = []

        if use_first_row_as_header and rows:
            headers = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T

        except ValueError as e:
            context['error'] = str(e)
            context['graph'] = None
            context['results'] = None
            return render(request, "descriptive/descriptive.html", context)

        max_significant_figures = max(count_significant_figures(num) for col in data_columns for num in col if not np.isnan(num))
        significant_figures = max_significant_figures + 2
        format_str = "{:." + str(significant_figures) + "g}"

        results = []
        z_value = 1.96
        means = []
        confidence_intervals = []
        coefficient_variations = []
        normality_results = []
        distribution_shape_results = []
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]

            if len(valid_col) > 0:
                mean = np.mean(valid_col)
                n_elements = len(valid_col)
                variance = np.var(valid_col, ddof=1)
                std_dev = np.std(valid_col, ddof=1)
                median = np.median(valid_col)
                std_error = std_dev / np.sqrt(n_elements)
                margin_of_error = z_value * std_error

                if np.isclose(std_error, 0.0):
                    confidence_interval = (mean, mean)
                else:
                    confidence_interval = stats.norm.interval(
                        0.95,
                        loc=mean,
                        scale=std_error
                    )

                confidence_interval_str = (
                    f"{format_str.format(confidence_interval[0])} ; "
                    f"{format_str.format(confidence_interval[1])}"
                )

                minimum = np.min(valid_col)
                maximum = np.max(valid_col)
                range_value = maximum - minimum
                percentile_25 = np.percentile(valid_col, 25)
                percentile_75 = np.percentile(valid_col, 75)
                interquartile_range = percentile_75 - percentile_25
                coefficient_variation = (
                    None
                    if np.isclose(mean, 0.0)
                    else (std_dev / abs(mean)) * 100
                )
                if np.isclose(std_dev, 0.0):
                    skewness = None
                    kurtosis = None
                else:
                    skewness = stats.skew(valid_col, bias=False)
                    kurtosis = stats.kurtosis(valid_col, bias=False)
                
                if np.isclose(std_dev, 0.0):
                    shapiro_w = None
                    shapiro_p = None
                else:
                    shapiro_w, shapiro_p = stats.shapiro(valid_col)

                variable_name = headers[i] if use_first_row_as_header and i < len(headers) else f"var. {i + 1}"

                results.append({
                    'variable': variable_name,
                    'n_elements': n_elements,
                    'minimum': format_str.format(minimum),
                    'maximum': format_str.format(maximum),
                    'range': format_str.format(range_value),
                    'percentile_25': format_str.format(percentile_25),
                    'median': format_str.format(median),
                    'percentile_75': format_str.format(percentile_75),
                    'iqr': format_str.format(interquartile_range),
                    'mean': format_str.format(mean),
                    'variance': format_str.format(variance),
                    'std_dev': format_str.format(std_dev),
                    'coefficient_variation': (
                        'N/A'
                        if coefficient_variation is None
                        else f"{format_str.format(coefficient_variation)}%"
                    ),
                    'std_error': format_str.format(std_error),
                    'margin_of_error': format_str.format(margin_of_error),
                    'confidence_interval': confidence_interval_str,
                    'skewness': (
                        'N/A'
                        if skewness is None
                        else "{:.4g}".format(skewness)
                    ),
                    'kurtosis': (
                        'N/A'
                        if kurtosis is None
                        else "{:.4g}".format(kurtosis)
                    ),
                    'shapiro_w': (
                        'N/A'
                        if shapiro_w is None
                        else "{:.4g}".format(shapiro_w)
                    ),
                    'shapiro_p': (
                        'N/A'
                        if shapiro_p is None
                        else "{:.4g}".format(shapiro_p)
                    ),
                })

                means.append(mean)
                confidence_intervals.append((confidence_interval[0], confidence_interval[1]))
                coefficient_variations.append(
                    (variable_name, coefficient_variation)
                )
                normality_results.append(
                    (variable_name, shapiro_w, shapiro_p)
                )
                distribution_shape_results.append(
                    (variable_name, skewness, kurtosis)
                )

        context['results'] = results
        context['data'] = data
        context['headers'] = headers if use_first_row_as_header else None
        context['use_first_row_as_header'] = 'checked' if use_first_row_as_header else ''

        request.session['results'] = results
        request.session['data'] = data
        request.session['headers'] = headers if use_first_row_as_header else None
        request.session['use_first_row_as_header'] = use_first_row_as_header

        # Mean + 95% CI Plotly chart with switch: Bars / Intervals
        variable_names = [result['variable'] for result in results]

        lower_errors = [
            mean - ci_lower
            for mean, (ci_lower, ci_upper)
            in zip(means, confidence_intervals)
        ]

        upper_errors = [
            ci_upper - mean
            for mean, (ci_lower, ci_upper)
            in zip(means, confidence_intervals)
        ]

        hover_data = [
            [
                result['n_elements'],
                result['confidence_interval'],
                result['std_dev'],
            ]
            for result in results
        ]

        chart_height = max(
            420,
            230 + (len(variable_names) * 30)
        )

        fig = go.Figure()

        # View 1: traditional vertical bars with 95% CI
        fig.add_trace(
            go.Bar(
                x=variable_names,
                y=means,

                width=0.38,

                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=upper_errors,
                    arrayminus=lower_errors,
                    thickness=1.6,
                    width=5,
                    visible=True,
                ),

                customdata=hover_data,

                hovertemplate=(
                    "<b>%{x}</b>"
                    "<br>Mean: %{y:.5g}"
                    "<br>95% CI: %{customdata[1]}"
                    "<br>Std. Dev.: %{customdata[2]}"
                    "<br>n: %{customdata[0]}"
                    "<extra></extra>"
                ),

                name='Bars',
                visible=True,
            )
        )

        # View 2: interval plot (forest-style)
        fig.add_trace(
            go.Scatter(
                x=means,
                y=variable_names,

                mode='markers',

                marker=dict(
                    size=10,
                ),

                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=upper_errors,
                    arrayminus=lower_errors,
                    thickness=1.6,
                    width=5,
                    visible=True,
                ),

                customdata=hover_data,

                hovertemplate=(
                    "<b>%{y}</b>"
                    "<br>Mean: %{x:.5g}"
                    "<br>95% CI: %{customdata[1]}"
                    "<br>Std. Dev.: %{customdata[2]}"
                    "<br>n: %{customdata[0]}"
                    "<extra></extra>"
                ),

                name='Intervals',
                visible=False,
            )
        )

        fig.update_layout(
            template='plotly_white',
            height=chart_height,

            margin=dict(
                l=45,
                r=30,
                t=80,
                b=60,
            ),

            showlegend=False,
            hovermode='closest',

            updatemenus=[
                dict(
                    type='buttons',
                    direction='right',

                    # Left side, away from Plotly modebar
                    x=0,
                    y=1.16,
                    xanchor='left',
                    yanchor='top',

                    showactive=True,

                    bgcolor='#ffffff',
                    bordercolor='#cbd5e1',
                    borderwidth=1,

                    pad=dict(
                        l=2,
                        r=2,
                        t=2,
                        b=2,
                    ),

                    font=dict(
                        size=12,
                    ),

                    buttons=[
                        dict(
                            label='Bars',
                            method='update',
                            args=[
                                {
                                    'visible': [True, False]
                                },
                                {
                                    'xaxis': {
                                        'title': None,
                                        'type': 'category',
                                        'showgrid': False,
                                        'tickangle': 0,
                                        'automargin': True,
                                    },
                                    'yaxis': {
                                        'title': 'Mean',
                                        'type': 'linear',
                                        'showgrid': True,
                                        'zeroline': False,
                                        'automargin': True,
                                        'autorange': True,
                                        'rangemode': 'tozero',
                                    },
                                },
                            ],
                        ),

                        dict(
                            label='Intervals',
                            method='update',
                            args=[
                                {
                                    'visible': [False, True]
                                },
                                {
                                    'xaxis': {
                                        'title': 'Mean',
                                        'type': 'linear',
                                        'showgrid': True,
                                        'zeroline': False,
                                        'automargin': True,
                                        'autorange': True,
                                    },
                                    'yaxis': {
                                        'title': None,
                                        'type': 'category',
                                        'showgrid': False,
                                        'automargin': True,
                                        'autorange': 'reversed',
                                        'categoryorder': 'array',
                                        'categoryarray': variable_names,
                                    },
                                },
                            ],
                        ),
                    ],
                )
            ],

            # Initial view = Bars
            xaxis=dict(
                title=None,
                type='category',
                showgrid=False,
                tickangle=-25,
                automargin=True,
            ),

            yaxis=dict(
                title='Mean',
                type='linear',
                showgrid=True,
                zeroline=False,
                automargin=True,
            ),
        )

        context['graph'] = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            config={
                'responsive': True,
                'displaylogo': False,
                'scrollZoom': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'descriptive_mean_95_ci',
                    'scale': 2,
                },
            },
        )

        request.session['graph'] = context['graph']


        # ------------------------------------------------------------------
        # Coefficient of Variation chart
        # ------------------------------------------------------------------

        cv_items = sorted(
            coefficient_variations,
            key=lambda item: (
                item[1] is not None,
                item[1] if item[1] is not None else -1
            ),
            reverse=True,
        )

        cv_valid = [
            (name, value)
            for name, value in cv_items
            if value is not None
        ]

        cv_not_available = [
            name
            for name, value in cv_items
            if value is None
        ]

        cv_names = [
            name
            for name, value in cv_valid
        ]

        cv_values = [
            value
            for name, value in cv_valid
        ]

        # Build the stems of the lollipop chart
        stem_x = []
        stem_y = []

        for name, value in cv_valid:
            stem_x.extend([0, value, None])
            stem_y.extend([name, name, None])


        cv_fig = go.Figure()


        # Horizontal stems
        if cv_valid:
            cv_fig.add_trace(
                go.Scatter(
                    x=stem_x,
                    y=stem_y,
                    mode='lines',
                    line=dict(
                        width=2,
                    ),
                    hoverinfo='skip',
                    showlegend=False,
                )
            )


        # Lollipop markers
        if cv_valid:
            cv_fig.add_trace(
                go.Scatter(
                    x=cv_values,
                    y=cv_names,
                    mode='markers+text',

                    marker=dict(
                        size=10,
                    ),

                    text=[
                        f"{format_str.format(value)}%"
                        for value in cv_values
                    ],

                    textposition='middle right',
                    cliponaxis=False,

                    customdata=[
                        [name, value]
                        for name, value in cv_valid
                    ],

                    hovertemplate=(
                        "<b>%{customdata[0]}</b>"
                        "<br>Coefficient of Variation: "
                        "%{customdata[1]:.5g}%"
                        "<extra></extra>"
                    ),

                    showlegend=False,
                )
            )


        # Variables for which CV is not defined
        if cv_not_available:
            cv_fig.add_trace(
                go.Scatter(
                    x=[0] * len(cv_not_available),
                    y=cv_not_available,

                    mode='text',

                    text=['N/A'] * len(cv_not_available),

                    textposition='middle right',

                    hovertemplate=(
                        "<b>%{y}</b>"
                        "<br>Coefficient of Variation: N/A"
                        "<br>Mean is zero or approximately zero"
                        "<extra></extra>"
                    ),

                    showlegend=False,
                )
            )


        cv_all_names = cv_names + cv_not_available

        cv_chart_height = max(
            360,
            170 + (len(cv_all_names) * 32)
        )


        cv_fig.update_layout(
            template='plotly_white',

            height=cv_chart_height,

            margin=dict(
                l=40,
                r=90,
                t=30,
                b=60,
            ),

            showlegend=False,

            hovermode='closest',

            xaxis=dict(
                title='Coefficient of Variation (%)',
                showgrid=True,
                zeroline=True,
                rangemode='tozero',
                automargin=True,
            ),

            yaxis=dict(
                title=None,
                categoryorder='array',
                categoryarray=cv_all_names,
                autorange='reversed',
                automargin=True,
                showgrid=False,
            ),
        )


        context['cv_graph'] = cv_fig.to_html(
            full_html=False,

            # Plotly was already loaded by the Mean + CI chart
            include_plotlyjs=False,

            config={
                'responsive': True,
                'displaylogo': False,
                'scrollZoom': True,

                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'descriptive_coefficient_variation',
                    'scale': 2,
                },
            },
        )

        request.session['cv_graph'] = context['cv_graph']


        # ------------------------------------------------------------------
        # Shapiro-Wilk normality diagnostic
        # ------------------------------------------------------------------

        normality_valid = sorted(
            [
                (name, w_value, p_value)
                for name, w_value, p_value in normality_results
                if p_value is not None
            ],
            key=lambda item: item[2]
        )

        normality_not_available = [
            name
            for name, w_value, p_value in normality_results
            if p_value is None
        ]

        normality_names = [
            name
            for name, w_value, p_value in normality_valid
        ]

        normality_w_values = [
            w_value
            for name, w_value, p_value in normality_valid
        ]

        normality_p_values = [
            p_value
            for name, w_value, p_value in normality_valid
        ]


        normality_fig = go.Figure()


        # Valid Shapiro-Wilk results
        if normality_valid:
            normality_fig.add_trace(
                go.Scatter(
                    x=normality_p_values,
                    y=normality_names,

                    mode='markers+text',

                    marker=dict(
                        size=10,
                    ),

                    text=[
                        f"{p_value:.4g}"
                        for p_value in normality_p_values
                    ],

                    textposition='middle right',
                    cliponaxis=False,

                    customdata=[
                        [w_value, p_value]
                        for w_value, p_value in zip(
                            normality_w_values,
                            normality_p_values
                        )
                    ],

                    hovertemplate=(
                        "<b>%{y}</b>"
                        "<br>Shapiro-Wilk W: %{customdata[0]:.5g}"
                        "<br>p-value: %{customdata[1]:.5g}"
                        "<extra></extra>"
                    ),

                    showlegend=False,
                )
            )


        # Variables for which Shapiro-Wilk is not applicable
        if normality_not_available:
            normality_fig.add_trace(
                go.Scatter(
                    x=[1.03] * len(normality_not_available),
                    y=normality_not_available,

                    mode='text',

                    text=['N/A'] * len(normality_not_available),

                    textposition='middle center',

                    hovertemplate=(
                        "<b>%{y}</b>"
                        "<br>Shapiro-Wilk: N/A"
                        "<br>Constant variable"
                        "<extra></extra>"
                    ),

                    showlegend=False,
                )
            )


        normality_all_names = (
            normality_names
            + normality_not_available
        )

        normality_chart_height = max(
            360,
            170 + (len(normality_all_names) * 32)
        )


        # Reference threshold alpha = 0.05
        normality_fig.add_vline(
            x=0.05,
            line_dash='dash',
            line_width=1.5,
            annotation_text='α = 0.05',
            annotation_position='top right',
        )


        normality_fig.update_layout(
            template='plotly_white',

            height=normality_chart_height,

            margin=dict(
                l=40,
                r=85,
                t=45,
                b=60,
            ),

            showlegend=False,

            hovermode='closest',

            xaxis=dict(
                title='Shapiro-Wilk p-value',
                range=[0, 1.08],
                showgrid=True,
                zeroline=False,
                automargin=True,
            ),

            yaxis=dict(
                title=None,
                categoryorder='array',
                categoryarray=normality_all_names,
                autorange='reversed',
                automargin=True,
                showgrid=False,
            ),
        )


        context['normality_graph'] = normality_fig.to_html(
            full_html=False,

            # Plotly is already loaded by the first chart
            include_plotlyjs=False,

            config={
                'responsive': True,
                'displaylogo': False,
                'scrollZoom': True,

                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'descriptive_shapiro_wilk',
                    'scale': 2,
                },
            },
        )

        request.session['normality_graph'] = context['normality_graph']


        # ------------------------------------------------------------------
        # Skewness vs. Excess Kurtosis
        # ------------------------------------------------------------------

        shape_valid = []
        shape_not_available = []

        for name, skew_value, kurt_value in distribution_shape_results:

            if (
                skew_value is None
                or kurt_value is None
                or not np.isfinite(skew_value)
                or not np.isfinite(kurt_value)
            ):
                shape_not_available.append(name)
            else:
                shape_valid.append(
                    (name, float(skew_value), float(kurt_value))
                )


        # Group variables that occupy the same point.
        # This avoids hiding coincident observations.
        shape_groups = {}

        for name, skew_value, kurt_value in shape_valid:

            key = (
                round(skew_value, 10),
                round(kurt_value, 10),
            )

            if key not in shape_groups:
                shape_groups[key] = {
                    'skewness': skew_value,
                    'kurtosis': kurt_value,
                    'variables': [],
                }

            shape_groups[key]['variables'].append(name)


        shape_points = list(shape_groups.values())

        shape_x = [
            point['skewness']
            for point in shape_points
        ]

        shape_y = [
            point['kurtosis']
            for point in shape_points
        ]

        shape_labels = [
            (
                point['variables'][0]
                if len(point['variables']) == 1
                else f"{len(point['variables'])} variables"
            )
            for point in shape_points
        ]

        shape_hover = [
            [
                ', '.join(point['variables']),
                len(point['variables']),
            ]
            for point in shape_points
        ]

        shape_marker_sizes = [
            min(
                22,
                10 + (len(point['variables']) - 1) * 3
            )
            for point in shape_points
        ]


        shape_fig = go.Figure()


        if shape_points:

            shape_fig.add_trace(
                go.Scatter(
                    x=shape_x,
                    y=shape_y,

                    mode='markers+text',

                    marker=dict(
                        size=shape_marker_sizes,
                        line=dict(
                            width=1,
                        ),
                    ),

                    text=shape_labels,
                    textposition='top center',

                    customdata=shape_hover,

                    hovertemplate=(
                        "<b>%{customdata[0]}</b>"
                        "<br>Skewness: %{x:.5g}"
                        "<br>Excess Kurtosis: %{y:.5g}"
                        "<br>Variables at this point: %{customdata[1]}"
                        "<extra></extra>"
                    ),

                    showlegend=False,
                )
            )

        else:

            shape_fig.add_annotation(
                x=0.5,
                y=0.5,
                xref='paper',
                yref='paper',
                text='No valid skewness / kurtosis values available',
                showarrow=False,
            )


        # Reference lines:
        # skewness = 0 -> symmetry reference
        # excess kurtosis = 0 -> normal-distribution reference
        shape_fig.add_vline(
            x=0,
            line_dash='dash',
            line_width=1.3,
        )

        shape_fig.add_hline(
            y=0,
            line_dash='dash',
            line_width=1.3,
        )


        shape_fig.update_layout(
            template='plotly_white',

            height=440,

            margin=dict(
                l=55,
                r=45,
                t=35,
                b=65,
            ),

            showlegend=False,

            hovermode='closest',

            xaxis=dict(
                title='Skewness',
                showgrid=True,
                zeroline=False,
                rangemode='tozero',
                automargin=True,
            ),

            yaxis=dict(
                title='Excess Kurtosis',
                showgrid=True,
                zeroline=False,
                rangemode='tozero',
                automargin=True,
            ),
        )


        # Mention variables for which shape measures are undefined
        if shape_not_available:

            shape_fig.add_annotation(
                x=0,
                y=-0.18,
                xref='paper',
                yref='paper',

                text=(
                    'Not shown (N/A): '
                    + ', '.join(shape_not_available)
                ),

                showarrow=False,
                xanchor='left',

                font=dict(
                    size=11,
                ),
            )


        context['shape_graph'] = shape_fig.to_html(
            full_html=False,

            # Plotly has already been loaded by the first chart
            include_plotlyjs=False,

            config={
                'responsive': True,
                'displaylogo': False,
                'scrollZoom': True,

                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'descriptive_skewness_kurtosis',
                    'scale': 2,
                },
            },
        )

        request.session['shape_graph'] = context['shape_graph']

##########################################################################################################################
    #############################################################################################################################
    # HISTOGRAMS
    #############################################################################################################################

    if request.method == "POST" and tab == "histograms":

        data = request.POST.get('data', '')
        use_first_row_as_header = (
            request.POST.get('use_first_row_as_header') == 'on'
        )

        hist_action = request.POST.get(
            'hist_action',
            'calculate'
        )

        if not data.strip():
            context['error'] = (
                "Please enter data to generate histogram/s."
            )
            context['graph_h'] = None

            return render(
                request,
                "descriptive/descriptive.html",
                context
            )

        data = data.replace('\r', '').strip()

        rows = [
            row.split('\t')
            for row in data.split('\n')
        ]

        columns = []
        headers = []

        if use_first_row_as_header and rows:
            headers = rows[0]
            rows = rows[1:]

        try:

            for row_idx, row in enumerate(rows):

                float_row = []

                for value in row:

                    value = value.strip()

                    if not value:

                        float_row.append(np.nan)

                    else:

                        try:
                            float_row.append(
                                float(value)
                            )

                        except ValueError:

                            if (
                                row_idx == 0
                                and not use_first_row_as_header
                            ):

                                raise ValueError(
                                    "The first row seems to contain "
                                    "non-numeric values, but "
                                    "'Use first row as header' "
                                    "is not checked."
                                )

                            raise ValueError(
                                "Non-numeric value found. "
                                "Please make sure all data entries "
                                "are valid numbers."
                            )

                columns.append(float_row)

            max_len = max(
                len(row)
                for row in columns
            )

            for i in range(len(columns)):

                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(
                columns
            ).T

        except ValueError as e:

            context['error'] = str(e)
            context['graph_h'] = None

            return render(
                request,
                "descriptive/descriptive.html",
                context
            )


        # -------------------------------------------------------------
        # Store entered data
        # -------------------------------------------------------------

        context['data'] = data

        context['headers'] = (
            headers
            if use_first_row_as_header
            else None
        )

        context['use_first_row_as_header'] = (
            'checked'
            if use_first_row_as_header
            else ''
        )

        request.session['data'] = data

        request.session['headers'] = (
            headers
            if use_first_row_as_header
            else None
        )

        request.session['use_first_row_as_header'] = (
            use_first_row_as_header
        )


        # -------------------------------------------------------------
        # Build list of valid variables
        # -------------------------------------------------------------

        histogram_columns = []

        for i, col in enumerate(data_columns):

            valid_col = col[
                ~np.isnan(col)
            ]

            if len(valid_col) == 0:
                continue

            variable_name = (
                headers[i]
                if use_first_row_as_header
                and i < len(headers)
                else f"var. {i + 1}"
            )

            histogram_columns.append(
                {
                    'index': str(i),
                    'name': variable_name,
                    'data': valid_col,
                }
            )


        if not histogram_columns:

            context['error'] = (
                "No valid numeric data available "
                "to generate histograms."
            )

            context['graph_h'] = None

            return render(
                request,
                "descriptive/descriptive.html",
                context
            )


        histogram_variables = [
            {
                'index': item['index'],
                'name': item['name'],
            }
            for item in histogram_columns
        ]


        valid_indices = [
            item['index']
            for item in histogram_columns
        ]


        # -------------------------------------------------------------
        # Per-variable histogram settings
        # -------------------------------------------------------------

        histogram_settings = request.session.get(
            'histogram_settings',
            {}
        )


        selected_variable = request.POST.get(
            'hist_variable'
        )

        if selected_variable is None:

            selected_variable = request.session.get(
                'histogram_selected_variable'
            )


        # Calculate with new data:
        # reset per-variable settings and select first valid variable.
        if hist_action == 'calculate':

            histogram_settings = {}

            selected_variable = (
                histogram_columns[0]['index']
            )


        if selected_variable not in valid_indices:

            selected_variable = (
                histogram_columns[0]['index']
            )


        # -------------------------------------------------------------
        # Available methods
        # -------------------------------------------------------------

        bin_method_labels = {
            'auto': 'Automatic',
            'fd': 'Freedman-Diaconis',
            'sturges': 'Sturges',
            'scott': 'Scott',
            'doane': 'Doane',
            'rice': 'Rice',
            'sqrt': 'Square root',
            'custom': 'Custom number of intervals',
        }


        # -------------------------------------------------------------
        # Reset selected variable only
        # -------------------------------------------------------------

        if hist_action == 'reset_bins':

            histogram_settings[
                selected_variable
            ] = {
                'method': 'auto',
                'custom_bins': None,
            }


        # -------------------------------------------------------------
        # Apply settings only to selected variable
        # -------------------------------------------------------------

        elif hist_action == 'apply_bins':

            selected_method = request.POST.get(
                'bin_method',
                'auto'
            )

            if selected_method not in bin_method_labels:

                selected_method = 'auto'


            custom_bins = None

            if selected_method == 'custom':

                try:

                    custom_bins = int(
                        request.POST.get(
                            'custom_bins',
                            ''
                        )
                    )

                except (TypeError, ValueError):

                    context['error'] = (
                        "Please enter a valid number "
                        "of intervals."
                    )

                    return render(
                        request,
                        "descriptive/descriptive.html",
                        context
                    )


                if not 1 <= custom_bins <= 100:

                    context['error'] = (
                        "The custom number of intervals "
                        "must be between 1 and 100."
                    )

                    return render(
                        request,
                        "descriptive/descriptive.html",
                        context
                    )


            histogram_settings[
                selected_variable
            ] = {
                'method': selected_method,
                'custom_bins': custom_bins,
            }


        # -------------------------------------------------------------
        # Current setting for selected variable
        # -------------------------------------------------------------

        current_setting = histogram_settings.get(
            selected_variable,
            {
                'method': 'auto',
                'custom_bins': None,
            }
        )


        current_method = current_setting.get(
            'method',
            'auto'
        )

        current_custom_bins = current_setting.get(
            'custom_bins'
        )


        if current_method not in bin_method_labels:

            current_method = 'auto'
            current_custom_bins = None


        # -------------------------------------------------------------
        # Get selected variable data
        # -------------------------------------------------------------

        selected_item = next(
            item
            for item in histogram_columns
            if item['index'] == selected_variable
        )


        variable_name = selected_item['name']

        valid_col = selected_item['data']


        # -------------------------------------------------------------
        # Determine bin edges
        # -------------------------------------------------------------

        if current_method == 'custom':

            bins_definition = current_custom_bins

        else:

            bins_definition = current_method


        try:

            bin_edges = np.histogram_bin_edges(
                valid_col,
                bins=bins_definition
            )

        except (ValueError, TypeError):

            context['error'] = (
                "Unable to calculate histogram intervals "
                "with the selected method."
            )

            context['graph_h'] = None

            return render(
                request,
                "descriptive/descriptive.html",
                context
            )


        counts, _ = np.histogram(
            valid_col,
            bins=bin_edges
        )


        selected_bins = (
            len(bin_edges) - 1
        )


        bin_widths = np.diff(
            bin_edges
        )


        bin_centers = (
            bin_edges[:-1]
            + bin_edges[1:]
        ) / 2


        cumulative_counts = np.cumsum(
            counts
        )


        relative_frequency = (
            counts / len(valid_col)
        ) * 100


        cumulative_relative_frequency = (
            cumulative_counts / len(valid_col)
        ) * 100


        bin_bounds = [
            [
                bin_edges[i],
                bin_edges[i + 1],
            ]
            for i in range(
                len(bin_edges) - 1
            )
        ]


        # Ogive starts at the lower class boundary with 0%
        ogive_x = np.concatenate(
            (
                [bin_edges[0]],
                bin_edges[1:]
            )
        )


        ogive_y = np.concatenate(
            (
                [0],
                cumulative_relative_frequency
            )
        )


        # -------------------------------------------------------------
        # Plotly figure
        # -------------------------------------------------------------

        fig = make_subplots(
            rows=2,
            cols=2,

            subplot_titles=(
                'Frequency',
                'Cumulative Frequency',
                'Relative Frequency',
                'Cumulative Relative Frequency'
            ),

            horizontal_spacing=0.12,
            vertical_spacing=0.18,
        )


        # Frequency
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=counts,
                width=bin_widths,

                customdata=bin_bounds,

                hovertemplate=(
                    "Interval: %{customdata[0]:.5g}"
                    " – %{customdata[1]:.5g}"
                    "<br>Frequency: %{y}"
                    "<extra></extra>"
                ),

                showlegend=False,
            ),
            row=1,
            col=1,
        )


        # Cumulative frequency
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=cumulative_counts,
                width=bin_widths,

                customdata=bin_bounds,

                hovertemplate=(
                    "Interval: %{customdata[0]:.5g}"
                    " – %{customdata[1]:.5g}"
                    "<br>Cumulative frequency: %{y}"
                    "<extra></extra>"
                ),

                showlegend=False,
            ),
            row=1,
            col=2,
        )


        # Relative frequency
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=relative_frequency,
                width=bin_widths,

                customdata=bin_bounds,

                hovertemplate=(
                    "Interval: %{customdata[0]:.5g}"
                    " – %{customdata[1]:.5g}"
                    "<br>Relative frequency: %{y:.2f}%"
                    "<extra></extra>"
                ),

                showlegend=False,
            ),
            row=2,
            col=1,
        )


        # Ogive
        fig.add_trace(
            go.Scatter(
                x=ogive_x,
                y=ogive_y,

                mode='lines+markers',

                marker=dict(
                    size=7,
                ),

                line=dict(
                    width=2,
                ),

                hovertemplate=(
                    "Value: %{x:.5g}"
                    "<br>Cumulative relative frequency: "
                    "%{y:.2f}%"
                    "<extra></extra>"
                ),

                showlegend=False,
            ),
            row=2,
            col=2,
        )


        fig.update_layout(
            template='plotly_white',

            title=dict(
                text=f"Distribution — {variable_name}",
                x=0,
                xanchor='left',

                font=dict(
                    size=16,
                ),
            ),

            height=680,

            margin=dict(
                l=55,
                r=35,
                t=75,
                b=60,
            ),

            showlegend=False,

            hovermode='closest',

            bargap=0.02,
        )


        # Axis labels

        fig.update_xaxes(
            title_text='Values',
            automargin=True,
            row=1,
            col=1,
        )

        fig.update_yaxes(
            title_text='Frequency',
            rangemode='tozero',
            automargin=True,
            row=1,
            col=1,
        )


        fig.update_xaxes(
            title_text='Values',
            automargin=True,
            row=1,
            col=2,
        )

        fig.update_yaxes(
            title_text='Cumulative Frequency',
            rangemode='tozero',
            automargin=True,
            row=1,
            col=2,
        )


        fig.update_xaxes(
            title_text='Values',
            automargin=True,
            row=2,
            col=1,
        )

        fig.update_yaxes(
            title_text='Relative Frequency (%)',
            rangemode='tozero',
            automargin=True,
            row=2,
            col=1,
        )


        fig.update_xaxes(
            title_text='Values',
            automargin=True,
            row=2,
            col=2,
        )

        fig.update_yaxes(
            title_text='Cumulative Relative Frequency (%)',
            range=[0, 105],
            automargin=True,
            row=2,
            col=2,
        )


        context['graph_h'] = fig.to_html(
            full_html=False,

            # Histograms is an independent tab,
            # so load Plotly here.
            include_plotlyjs='cdn',

            config={
                'responsive': True,
                'displaylogo': False,
                'scrollZoom': True,

                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': (
                        'descriptive_histogram'
                    ),
                    'scale': 2,
                },
            },
        )


        # -------------------------------------------------------------
        # Context and session
        # -------------------------------------------------------------

        method_label = bin_method_labels[
            current_method
        ]


        context['histogram_variables'] = (
            histogram_variables
        )

        context['histogram_selected_variable'] = (
            selected_variable
        )

        context['histogram_bin_method'] = (
            current_method
        )

        context['histogram_custom_bins'] = (
            current_custom_bins
            if current_custom_bins is not None
            else ''
        )

        context['histogram_selected_bins'] = (
            selected_bins
        )

        context['histogram_method_label'] = (
            method_label
        )


        request.session['histogram_settings'] = (
            histogram_settings
        )

        request.session['histogram_variables'] = (
            histogram_variables
        )

        request.session[
            'histogram_selected_variable'
        ] = selected_variable

        request.session[
            'histogram_bin_method'
        ] = current_method

        request.session[
            'histogram_custom_bins'
        ] = (
            current_custom_bins
            if current_custom_bins is not None
            else ''
        )

        request.session[
            'histogram_selected_bins'
        ] = selected_bins

        request.session[
            'histogram_method_label'
        ] = method_label

        request.session['graph_h'] = (
            context['graph_h']
        )

#############################################################################################################################    
    if request.method == "POST" and tab == "boxplot":
        data = request.POST.get('data')
        use_first_row_as_header = request.POST.get('use_first_row_as_header') == 'on'

        if not data.strip():
            context['error'] = "Please enter data to generate boxplot/s."
            context['boxplot'] = None
            return render(request, "descriptive/descriptive.html", context)
        
        data = data.replace('\r', '').strip()
        rows = [row.split('\t') for row in data.split('\n')]
        columns = []
        headers = []

        if use_first_row_as_header and rows:
            headers = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        float_row.append(np.nan)
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)

            max_len = max(len(row) for row in columns)

            for i in range(len(columns)):
                while len(columns[i]) < max_len:
                    columns[i].append(np.nan)

            data_columns = np.array(columns).T
        
        except ValueError as e:
            context['error'] = str(e)
            context['boxplot'] = None
            return render(request, "descriptive/descriptive.html", context)

        context['data'] = data
        context['headers'] = headers if use_first_row_as_header else None
        context['use_first_row_as_header'] = 'checked' if use_first_row_as_header else ''

        request.session['data'] = data
        request.session['headers'] = headers if use_first_row_as_header else None
        request.session['use_first_row_as_header'] = use_first_row_as_header

        fig, ax = plt.subplots()
        for i, col in enumerate(data_columns):
            valid_col = col[~np.isnan(col)]

            if len(valid_col) > 0:
                ax.boxplot(valid_col, positions=[i + 1], widths=0.5)

        ax.set_title('Boxplot')
        ax.set_ylabel('Values')
        ax.set_xticks(range(1, len(data_columns) + 1))
        ax.set_xticklabels(headers if use_first_row_as_header else [f"var. {i + 1}" for i in range(len(data_columns))])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        boxplot_data = base64.b64encode(buf.read()).decode('utf-8')
        context['boxplot'] = f'data:image/png;base64,{boxplot_data}'
        request.session['boxplot'] = context['boxplot']


    return render(request, "descriptive/descriptive.html", context)


def count_significant_figures(num_str):
    if isinstance(num_str, (float, np.float64)):
        num_str = str(num_str)
    else:
        num_str = num_str
    
    num_str = num_str.strip()
    num_str = re.sub(r'[eE][+-]?\d+', '', num_str)
    num_str = num_str.replace('.', '')
    return len(num_str.lstrip('0'))