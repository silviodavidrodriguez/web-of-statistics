from django.shortcuts import render
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import plotly.graph_objects as go
from scipy.optimize import curve_fit

def regression(request):
    tab = request.GET.get('tab', None)
    subtab = request.GET.get('subtab', None)

    context = {
        'segment': 'regression',
        'active_tab': tab,
        'active_subtab': subtab,
        'data_lin_reg': request.session.get('data_lin_reg', ""),
        'results_lin_reg': request.session.get('results_lin_reg', None),
        'outliers': request.session.get('outliers', None),
        'reg_graph_lin_reg_2D': request.session.get('reg_graph_lin_reg_2D', None),
        'reg_graph_lin_reg_2D_no_conf': request.session.get('reg_graph_lin_reg_2D_no_conf', None),
        'reg_graph_lin_reg_3D': request.session.get('reg_graph_lin_reg_3D', None),
        'reg_graph_lin_reg_3D_no_conf': request.session.get('reg_graph_lin_reg_3D_no_conf', None),
        'residuals_plots': request.session.get('residuals_plots', None),
        'headers_lin_reg': request.session.get('headers_lin_reg', None),
        'use_first_row_as_header_lin_reg': 'checked' if request.session.get('use_first_row_as_header_lin_reg', False) else '',
        'model_description_lin_reg': request.session.get('model_description_lin_reg', None),
        'data_lin_reg_int': request.session.get('data_lin_reg_int', ""),
        'results_lin_reg_int': request.session.get('results_lin_reg_int', None),
        'outliers_int': request.session.get('outliers_int', None),
        'reg_graph_lin_reg_int_3D': request.session.get('reg_graph_lin_reg_int_3D', None),
        'reg_graph_lin_reg_int_3D_no_conf': request.session.get('reg_graph_lin_reg_int_3D_no_conf', None),
        'residuals_plots_int': request.session.get('residuals_plots_int', None),
        'headers_lin_reg_int': request.session.get('headers_lin_reg_int', None),
        'use_first_row_as_header_lin_reg_int': 'checked' if request.session.get('use_first_row_as_header_lin_reg_int', False) else '',
        'model_description_lin_reg_int': request.session.get('model_description_lin_reg_int', None),
        'data_nonlin_reg': request.session.get('data_nonlin_reg', ""),
        'results_nonlin_reg': request.session.get('results_nonlin_reg', None),
        'nonlin_graph': request.session.get('nonlin_graph', None),
        'nonlin_graph_no_conf': request.session.get('nonlin_graph_no_conf', None),
        'headers_nonlin_reg': request.session.get('headers_nonlin_reg', None),
        'use_first_row_as_header_nonlin_reg': 'checked' if request.session.get('use_first_row_as_header_nonlin_reg', False) else '',
        'nonlinear_function': request.session.get('nonlinear_function', None),
    }

    if request.method == "POST" and request.POST.get("clear_lin_reg") == "true":
        if 'data_lin_reg' in request.session:
            del request.session['data_lin_reg']
        if 'results_lin_reg' in request.session:
            del request.session['results_lin_reg']
        if 'outliers' in request.session:
            del request.session['outliers']
        if 'reg_graph_lin_reg_2D' in request.session:
            del request.session['reg_graph_lin_reg_2D']
        if 'reg_graph_lin_reg_2D_no_conf' in request.session:
            del request.session['reg_graph_lin_reg_2D_no_conf']
        if 'reg_graph_lin_reg_3D' in request.session:
            del request.session['reg_graph_lin_reg_3D']
        if 'reg_graph_lin_reg_3D_no_conf' in request.session:
            del request.session['reg_graph_lin_reg_3D_no_conf']
        if 'residuals_plots' in request.session:
            del request.session['residuals_plots']
        if 'headers_lin_reg' in request.session:
            del request.session['headers_lin_reg']
        if 'use_first_row_as_header_lin_reg' in request.session:
            request.session.pop('use_first_row_as_header_lin_reg', False)
        if 'model_description_lin_reg' in request.session:
            del request.session['model_description_lin_reg']
        context['data_lin_reg'] = ""
        context['results_lin_reg'] = None
        context['outliers'] = None
        context['reg_graph_lin_reg_2D'] = None
        context['reg_graph_lin_reg_2D_no_conf'] = None
        context['reg_graph_lin_reg_3D'] = None
        context['reg_graph_lin_reg_3D_no_conf'] = None
        context['residuals_plots'] = None
        context['model_description_lin_reg'] = None
        context['headers_lin_reg'] = None
        context['use_first_row_as_header_lin_reg'] = False
        return render(request, "regression/regression.html", context)
    
    if request.method == "POST" and request.POST.get("clear_lin_reg_int") == "true":
        if 'data_lin_reg_int' in request.session:
            del request.session['data_lin_reg_int']
        if 'results_lin_reg_int' in request.session:
            del request.session['results_lin_reg_int']
        if 'outliers_int' in request.session:
            del request.session['outliers_int']
        if 'reg_graph_lin_reg_int_3D' in request.session:
            del request.session['reg_graph_lin_reg_int_3D']
        if 'reg_graph_lin_reg_int_3D_no_conf' in request.session:
            del request.session['reg_graph_lin_reg_int_3D_no_conf']
        if 'residuals_plots_int' in request.session:
            del request.session['residuals_plots_int']
        if 'headers_lin_reg_int' in request.session:
            del request.session['headers_lin_reg_int']
        if 'use_first_row_as_header_lin_reg_int' in request.session:
            request.session.pop('use_first_row_as_header_lin_reg_int', False)
        if 'model_description_lin_reg_int' in request.session:
            del request.session['model_description_lin_reg_int']
        context['data_lin_reg_int'] = ""
        context['results_lin_reg_int'] = None
        context['outliers_int'] = None
        context['reg_graph_lin_reg_int_3D'] = None
        context['reg_graph_lin_reg_int_3D_no_conf'] = None
        context['residuals_plots_int'] = None
        context['model_description_lin_reg_int'] = None
        context['headers_lin_reg_int'] = None
        context['use_first_row_as_header_lin_reg_int'] = False
        return render(request, "regression/regression.html", context)
    
    if request.method == "POST" and request.POST.get("clear_nonlin_reg") == "true":
        if 'data_nonlin_reg' in request.session:
            del request.session['data_nonlin_reg']
        if 'results_nonlin_reg' in request.session:
            del request.session['results_nonlin_reg']
        if 'nonlin_graph' in request.session:
            del request.session['nonlin_graph']
        if 'nonlin_graph_no_conf' in request.session:
            del request.session['nonlin_graph_no_conf']
        if 'headers_nonlin_reg' in request.session:
            del request.session['headers_nonlin_reg']
        if 'use_first_row_as_header_nonlin_reg' in request.session:
            request.session.pop('use_first_row_as_header_nonlin_reg', False)
        if 'nonlinear_function' in request.session:
            del request.session['nonlinear_function']
        context['data_nonlin_reg'] = ""
        context['results_nonlin_reg'] = None
        context['nonlin_graph'] = None
        context['nonlin_graph_no_conf'] = None
        context['headers_nonlin_reg'] = None
        context['use_first_row_as_header_nonlin_reg'] = False
        context['nonlinear_function'] = None
        return render(request, "regression/regression.html", context)

##################################################################################################
    if request.method == "POST" and tab == "lin_reg" and subtab == "lin_reg":
        data_lin_reg = request.POST.get('data_lin_reg')
        use_first_row_as_header_lin_reg = request.POST.get('use_first_row_as_header_lin_reg') == 'on'

        context['use_first_row_as_header_lin_reg'] = 'checked' if use_first_row_as_header_lin_reg else ''
        request.session['use_first_row_as_header_lin_reg'] = use_first_row_as_header_lin_reg
        context['data_lin_reg'] = data_lin_reg
        request.session['data_lin_reg'] = data_lin_reg

        if not data_lin_reg.strip():
            context['error_lin_reg'] = "Please enter data before calculating."
            context['results_lin_reg'] = None
            context['reg_graph_lin_reg_2D'] = None
            context['reg_graph_lin_reg_2D_no_conf'] = None
            context['reg_graph_lin_reg_3D'] = None
            context['reg_graph_lin_reg_3D_no_conf'] = None
            context['model_description_lin_reg'] = None
            context['residuals_plot'] = None
            context['outliers'] = None
            return render(request, "regression/regression.html", context)

        data_lin_reg = data_lin_reg.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_lin_reg.split('\n')]
        columns = []
        row_lengths = []
        headers_lin_reg = []

        if use_first_row_as_header_lin_reg and rows:
            headers_lin_reg = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        raise ValueError("Missing value/s detected. Please ensure all values are present.")
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_lin_reg:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)
                row_lengths.append(len(float_row))

            if len(set(row_lengths)) > 1:
                raise ValueError("Missing value/s detected. Please ensure all values are present.")

            data_columns = np.array(columns).T
            data_columns = data_columns.T

            if not headers_lin_reg:
                num_columns = data_columns.shape[1]
                if num_columns == 2:
                    headers_lin_reg = ["Y", "X1"]
                else:
                    headers_lin_reg = ["Y"] + [f"X{i}" for i in range(1, num_columns)]
            
            context['headers_lin_reg'] = headers_lin_reg
            request.session['headers_lin_reg'] = headers_lin_reg

            data_frame = pd.DataFrame(data_columns, columns=headers_lin_reg)

            Y = data_frame.iloc[:, 0]
            X = data_frame.iloc[:, 1:]
            X = sm.add_constant(X)

            model = sm.OLS(Y, X)
            results = model.fit()

            fitted_values = results.fittedvalues
            predictions = results.get_prediction(X)
            conf_int = predictions.conf_int(alpha=0.05)
            lower_bound = conf_int[:, 0]
            upper_bound = conf_int[:, 1]    
            
            conf_int_r = results.conf_int(alpha=0.05)
            param_names = ['β0'] + [f'β{i}' for i in range(1, len(results.params))]

            params_with_values = list(zip(
                param_names,
                results.params.values,
                results.bse,
                results.tvalues,
                results.pvalues,
                conf_int_r.iloc[:, 0].values,
                conf_int_r.iloc[:, 1].values,
            ))

            context['results_lin_reg'] = {
                "R_squared": results.rsquared,
                "Adj_R_squared": results.rsquared_adj,
                "params": params_with_values,
                "f_statistic": results.fvalue,
                "f_pvalue": results.f_pvalue,
            }

            request.session['results_lin_reg'] = context['results_lin_reg']

            beta_values = [f"β{i}" for i in range(len(results.params))]
            model_description_lin_reg = f"{headers_lin_reg[0]} = {beta_values[0]}"

            for idx, header in enumerate(headers_lin_reg[1:], start=1):
                model_description_lin_reg += f" + ({beta_values[idx]}.{header})"

            context['model_description_lin_reg'] = model_description_lin_reg
            request.session['model_description_lin_reg'] = model_description_lin_reg

            residuals = results.resid
            std_residuals = residuals / np.std(residuals)

            if X.shape[1] == 2:
                sorted_indices = np.argsort(X.iloc[:, 1])
                X_sorted = X.iloc[sorted_indices, 1]
                lower_bound_sorted = lower_bound[sorted_indices]
                upper_bound_sorted = upper_bound[sorted_indices]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=X.iloc[:, 1], y=Y, mode='markers', name='Data Points', 
                    marker=dict(color='blue'), showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=X.iloc[:, 1], y=fitted_values, mode='lines', name='Fitted Line', 
                    line=dict(color='red'), showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=X_sorted, y=lower_bound_sorted, fill=None, mode='lines', 
                    name='Lower Confidence Band', line=dict(color='lightgray'), showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=X_sorted, y=upper_bound_sorted, fill='tonexty', mode='lines', 
                    name='Upper Confidence Band', line=dict(color='lightgray'), showlegend=False
                ))

                fig.update_layout(
                    title="Regression Plot with 95% Confidence Band",
                    title_x=0.5,
                    plot_bgcolor="white",
                    xaxis=dict(
                        title=headers_lin_reg[1],
                        showgrid=True,
                        zeroline=False,
                        showline=True,
                        linecolor='black',
                        tickangle=0,
                        ticks='inside',
                        tickcolor='black',
                    ),
                    yaxis=dict(
                        title=headers_lin_reg[0],
                        showgrid=True,
                        zeroline=False,
                        showline=True,
                        linecolor='black',
                        ticks='inside',
                        tickcolor='black',
                    ),
                    template="plotly",
                    showlegend=False
                )

                reg_graph_lin_reg_2D = fig.to_html(full_html=False)
                context['reg_graph_lin_reg_2D'] = reg_graph_lin_reg_2D
                request.session['reg_graph_lin_reg_2D'] = reg_graph_lin_reg_2D


                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=X.iloc[:, 1], y=Y, mode='markers', name='Data Points', 
                    marker=dict(color='blue'), showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=X.iloc[:, 1], y=fitted_values, mode='lines', name='Fitted Line', 
                    line=dict(color='red'), showlegend=False
                ))

                fig.update_layout(
                    title="Regression Plot Without Confidence Band",
                    title_x=0.5,
                    plot_bgcolor="white",
                    xaxis=dict(
                        title=headers_lin_reg[1],
                        showgrid=True,
                        zeroline=False,
                        showline=True,
                        linecolor='black',
                        tickangle=0,
                        ticks='inside',
                        tickcolor='black',
                    ),
                    yaxis=dict(
                        title=headers_lin_reg[0],
                        showgrid=True,
                        zeroline=False,
                        showline=True,
                        linecolor='black',
                        ticks='inside',
                        tickcolor='black',
                    ),
                    template="plotly",
                    showlegend=False
                )

                reg_graph_lin_reg_2D_no_conf = fig.to_html(full_html=False)
                context['reg_graph_lin_reg_2D_no_conf'] = reg_graph_lin_reg_2D_no_conf
                request.session['reg_graph_lin_reg_2D_no_conf'] = reg_graph_lin_reg_2D_no_conf

            if X.shape[1] == 3:
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=X.iloc[:, 1], y=X.iloc[:, 2], z=Y,
                    mode='markers',
                    marker=dict(size=5, opacity=0.8, color='blue'),
                    name='Data Points'
                ))

                x_range = np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 10)
                y_range = np.linspace(X.iloc[:, 2].min(), X.iloc[:, 2].max(), 10)
                x_grid, y_grid = np.meshgrid(x_range, y_range)

                intercept = results.params[0]
                coef1 = results.params[1]
                coef2 = results.params[2]
                z_grid = intercept + coef1 * x_grid + coef2 * y_grid

                sorted_indices = np.argsort(X.iloc[:, 1])
                X_sorted = X.iloc[sorted_indices, 1]
                Y_sorted = Y.iloc[sorted_indices]
                lower_bound_sorted = lower_bound[sorted_indices]
                upper_bound_sorted = upper_bound[sorted_indices]
                lower_z_grid = intercept + coef1 * x_grid + coef2 * y_grid - (upper_bound_sorted - lower_bound_sorted).mean()
                upper_z_grid = intercept + coef1 * x_grid + coef2 * y_grid + (upper_bound_sorted - lower_bound_sorted).mean()

                fig.add_trace(go.Surface(
                    x=x_grid, y=y_grid, z=z_grid,
                    colorscale=[[0, '#F08080'], [1, '#F08080']],
                    showscale=False,
                    opacity=0.5,
                    name='Regression Plane'
                ))

                fig.add_trace(go.Surface(
                    x=x_grid, y=y_grid, z=lower_z_grid,
                    colorscale=[[0, 'lightgray'], [1, 'lightgray']],
                    showscale=False,
                    opacity=0.4,
                    name='Lower Confidence Band'
                ))

                fig.add_trace(go.Surface(
                    x=x_grid, y=y_grid, z=upper_z_grid,
                    colorscale=[[0, 'lightgray'], [1, 'lightgray']],
                    showscale=False,
                    opacity=0.4,
                    name='Upper Confidence Band'
                ))

                fig.update_layout(
                    scene=dict(
                        xaxis_title=headers_lin_reg[1],
                        yaxis_title=headers_lin_reg[2],
                        zaxis_title=headers_lin_reg[0],
                    ),
                    margin=dict(r=10, l=10, b=10, t=40)
                )

                reg_graph_lin_reg_3D = fig.to_html(full_html=False)
                context['reg_graph_lin_reg_3D'] = reg_graph_lin_reg_3D
                request.session['reg_graph_lin_reg_3D'] = reg_graph_lin_reg_3D

                fig_no_conf = go.Figure()
                fig_no_conf.add_trace(go.Scatter3d(
                    x=X.iloc[:, 1], y=X.iloc[:, 2], z=Y,
                    mode='markers',
                    marker=dict(size=5, opacity=0.8, color='blue'),
                    name='Data Points'
                ))

                fig_no_conf.add_trace(go.Surface(
                    x=x_grid, y=y_grid, z=z_grid,
                    colorscale=[[0, '#F08080'], [1, '#F08080']],
                    showscale=False,
                    opacity=0.5,
                    name='Regression Plane'
                ))

                fig_no_conf.update_layout(
                    scene=dict(
                        xaxis_title=headers_lin_reg[1],
                        yaxis_title=headers_lin_reg[2],
                        zaxis_title=headers_lin_reg[0],
                    ),
                    margin=dict(r=10, l=10, b=10, t=40)
                )

                reg_graph_lin_reg_3D_no_conf = fig_no_conf.to_html(full_html=False)
                context['reg_graph_lin_reg_3D_no_conf'] = reg_graph_lin_reg_3D_no_conf
                request.session['reg_graph_lin_reg_3D_no_conf'] = reg_graph_lin_reg_3D_no_conf

            residuals_plots = []
            for idx, header in enumerate(headers_lin_reg[1:], start=1):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=X.iloc[:, idx],
                    y=std_residuals,
                    mode='markers',
                    marker=dict(color="blue", opacity=0.7),
                    name='Residuals'
                ))

                fig.add_trace(go.Scatter(
                    x=[min(X.iloc[:, idx]), max(X.iloc[:, idx])],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color="black", dash="dash"),
                    name='Zero Line'
                ))

                fig.add_trace(go.Scatter(
                    x=[min(X.iloc[:, idx]), max(X.iloc[:, idx])],
                    y=[3, 3],
                    mode='lines',
                    line=dict(color="red", dash="dash"),
                    name='Upper Bound (+3)'
                ))

                fig.add_trace(go.Scatter(
                    x=[min(X.iloc[:, idx]), max(X.iloc[:, idx])],
                    y=[-3, -3],
                    mode='lines',
                    line=dict(color="red", dash="dash"),
                    name='Lower Bound (-3)'
                ))

                fig.update_layout(
                    title=f"Residuals vs {header}",
                    xaxis_title=header,
                    yaxis_title="Standardized Residuals",
                    showlegend=False,
                    template="plotly_white"
                )

                residual_plot_html = fig.to_html(full_html=False)
                residuals_plots.append(residual_plot_html)
            context['residuals_plots'] = residuals_plots
            request.session['residuals_plots'] = residuals_plots

            outliers = []
            for idx, residual in enumerate(std_residuals):
                if residual > 3 or residual < -3:
                    outlier_info = {
                        'index': idx+1,
                        'X_values': X.iloc[idx, 1:].tolist(),
                        'residual': round(residual, 4),
                        'actual_value': Y.iloc[idx],
                    }
                    outliers.append(outlier_info)
            context['outliers'] = outliers
            request.session['outliers'] = outliers

        except ValueError as e:
            context['error_lin_reg'] = str(e)
            context['results_lin_reg'] = None
            context['reg_graph_lin_reg_2D'] = None
            context['reg_graph_lin_reg_2D_no_conf'] = None
            context['reg_graph_lin_reg_3D'] = None
            context['reg_graph_lin_reg_3D_no_conf'] = None
            context['model_description_lin_reg'] = None
            context['residuals_plots'] = None
            context['outliers'] = None
            return render(request, "regression/regression.html", context)

###################################################################################################
    if request.method == "POST" and tab == "lin_reg" and subtab == "lin_reg_int":
        data_lin_reg_int = request.POST.get('data_lin_reg_int')
        use_first_row_as_header_lin_reg_int = request.POST.get('use_first_row_as_header_lin_reg_int') == 'on'

        context['use_first_row_as_header_lin_reg_int'] = 'checked' if use_first_row_as_header_lin_reg_int else ''
        request.session['use_first_row_as_header_lin_reg_int'] = use_first_row_as_header_lin_reg_int
        context['data_lin_reg_int'] = data_lin_reg_int
        request.session['data_lin_reg_int'] = data_lin_reg_int

        if not data_lin_reg_int.strip():
            context['error_lin_reg_int'] = "Please enter data before calculating."
            context['results_lin_reg_int'] = None
            context['reg_graph_lin_reg_int_3D'] = None
            context['reg_graph_lin_reg_int_3D_no_conf'] = None
            context['model_description_lin_reg_int'] = None
            context['residuals_plots_int'] = None
            context['outliers_int'] = None
            return render(request, "regression/regression.html", context)
        
        data_lin_reg_int = data_lin_reg_int.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_lin_reg_int.split('\n')]
        columns = []
        row_lengths = []
        headers_lin_reg_int = []

        if use_first_row_as_header_lin_reg_int and rows:
            headers_lin_reg_int = rows[0]
            rows = rows[1:]

        try:
            for row_idx, row in enumerate(rows):
                float_row = []
                for value in row:
                    value = value.strip()
                    if not value:
                        raise ValueError("Missing value/s detected. Please ensure all values are present.")
                    else:
                        try:
                            float_row.append(float(value))
                        except ValueError:
                            if row_idx == 0 and not use_first_row_as_header_lin_reg_int:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)
                row_lengths.append(len(float_row))

            if len(set(row_lengths)) > 1:
                raise ValueError("Missing value/s detected. Please ensure all values are present.")

            data_columns = np.array(columns).T
            data_columns = data_columns.T

            if not headers_lin_reg_int:
                num_columns = data_columns.shape[1]
                if num_columns == 2:
                    headers_lin_reg_int = ["Y", "X1"]
                else:
                    headers_lin_reg_int = ["Y"] + [f"X{i}" for i in range(1, num_columns)]

            context['headers_lin_reg_int'] = headers_lin_reg_int
            request.session['headers_lin_reg_int'] = headers_lin_reg_int

            data_frame = pd.DataFrame(data_columns, columns=headers_lin_reg_int)

            Y = data_frame.iloc[:, 0]
            X = data_frame.iloc[:, 1:]

            if X.shape[1] < 2:
                raise ValueError("At least two X variables are required to calculate interactions.")

            interaction_terms = []
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    interaction_terms.append(X.iloc[:, i] * X.iloc[:, j])
            
            X = pd.concat([X] + interaction_terms, axis=1)
            X = sm.add_constant(X)

            model = sm.OLS(Y, X)
            results = model.fit()

            fitted_values = results.fittedvalues
            predictions = results.get_prediction(X)
            conf_int = predictions.conf_int(alpha=0.05)
            lower_bound = conf_int[:, 0]
            upper_bound = conf_int[:, 1]

            conf_int_r = results.conf_int(alpha=0.05)
            param_names = ['β0'] + [f'β{i}' for i in range(1, len(results.params))]

            params_with_values = list(zip(
                param_names,
                results.params.values,
                results.bse,
                results.tvalues,
                results.pvalues,
                conf_int_r.iloc[:, 0].values,
                conf_int_r.iloc[:, 1].values,
            ))

            context['results_lin_reg_int'] = {
                "R_squared": results.rsquared,
                "Adj_R_squared": results.rsquared_adj,
                "params": params_with_values,
                "f_statistic": results.fvalue,
                "f_pvalue": results.f_pvalue,
            }

            request.session['results_lin_reg_int'] = context['results_lin_reg_int']

            interaction_names = []
            num_headers = len(headers_lin_reg_int)
            for i in range(num_headers - 2):
                for j in range(i + 1, num_headers - 1):
                    interaction_names.append(f"{headers_lin_reg_int[i + 1]}*{headers_lin_reg_int[j + 1]}")

            beta_values = [f"β{i}" for i in range(len(results.params))]
            model_description_lin_reg_int = f"{headers_lin_reg_int[0]} = {beta_values[0]}"

            for idx, header in enumerate(headers_lin_reg_int[1:], start=1):
                model_description_lin_reg_int += f" + ({beta_values[idx]} * {header})"

            for idx, interaction_name in enumerate(interaction_names, start=len(headers_lin_reg_int)):
                model_description_lin_reg_int += f" + ({beta_values[idx]} * {interaction_name})"

            context['model_description_lin_reg_int'] = model_description_lin_reg_int
            request.session['model_description_lin_reg_int'] = model_description_lin_reg_int

            residuals = results.resid
            std_residuals = residuals / np.std(residuals)

            if X.shape[1] == 4:
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=X.iloc[:, 1], y=X.iloc[:, 2], z=Y,
                    mode='markers',
                    marker=dict(size=5, opacity=0.8, color='blue'),
                    name='Data Points'
                ))

                x_range = np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 10)
                y_range = np.linspace(X.iloc[:, 2].min(), X.iloc[:, 2].max(), 10)
                x_grid, y_grid = np.meshgrid(x_range, y_range)

                intercept = results.params["const"]
                coef1 = results.params[headers_lin_reg_int[1]]
                coef2 = results.params[headers_lin_reg_int[2]]
                coef3 = results.params[0]
                z_grid = intercept + coef1 * x_grid + coef2 * y_grid + coef3 * (x_grid * y_grid)

                sorted_indices = np.argsort(X.iloc[:, 1])
                X_sorted = X.iloc[sorted_indices, 1]
                Y_sorted = Y.iloc[sorted_indices]
                lower_bound_sorted = lower_bound[sorted_indices]
                upper_bound_sorted = upper_bound[sorted_indices]
                lower_z_grid = intercept + coef1 * x_grid + coef2 * y_grid + coef3 * (x_grid * y_grid) - (upper_bound_sorted - lower_bound_sorted).mean()
                upper_z_grid = intercept + coef1 * x_grid + coef2 * y_grid + coef3 * (x_grid * y_grid) + (upper_bound_sorted - lower_bound_sorted).mean()

                fig.add_trace(go.Surface(
                    x=x_grid, y=y_grid, z=z_grid,
                    colorscale=[[0, '#F08080'], [1, '#F08080']],
                    showscale=False,
                    opacity=0.5,
                    name='Regression Plane'
                ))

                fig.add_trace(go.Surface(
                    x=x_grid, y=y_grid, z=lower_z_grid,
                    colorscale=[[0, 'lightgray'], [1, 'lightgray']],
                    showscale=False,
                    opacity=0.4,
                    name='Lower Confidence Band'
                ))

                fig.add_trace(go.Surface(
                    x=x_grid, y=y_grid, z=upper_z_grid,
                    colorscale=[[0, 'lightgray'], [1, 'lightgray']],
                    showscale=False,
                    opacity=0.4,
                    name='Upper Confidence Band'
                ))

                fig.update_layout(
                    scene=dict(
                        xaxis_title=headers_lin_reg_int[1],
                        yaxis_title=headers_lin_reg_int[2],
                        zaxis_title=headers_lin_reg_int[0],
                    ),
                    margin=dict(r=10, l=10, b=10, t=40)
                )

                reg_graph_lin_reg_int_3D = fig.to_html(full_html=False)
                context['reg_graph_lin_reg_int_3D'] = reg_graph_lin_reg_int_3D
                request.session['reg_graph_lin_reg_int_3D'] = reg_graph_lin_reg_int_3D

                fig_no_conf = go.Figure()
                fig_no_conf.add_trace(go.Scatter3d(
                    x=X.iloc[:, 1], y=X.iloc[:, 2], z=Y,
                    mode='markers',
                    marker=dict(size=5, opacity=0.8, color='blue'),
                    name='Data Points'
                ))

                fig_no_conf.add_trace(go.Surface(
                    x=x_grid, y=y_grid, z=z_grid,
                    colorscale=[[0, '#F08080'], [1, '#F08080']],
                    showscale=False,
                    opacity=0.5,
                    name='Regression Plane'
                ))

                fig_no_conf.update_layout(
                    scene=dict(
                        xaxis_title=headers_lin_reg_int[1],
                        yaxis_title=headers_lin_reg_int[2],
                        zaxis_title=headers_lin_reg_int[0],
                    ),
                    margin=dict(r=10, l=10, b=10, t=40)
                )

                reg_graph_lin_reg_int_3D_no_conf = fig_no_conf.to_html(full_html=False)
                context['reg_graph_lin_reg_int_3D_no_conf'] = reg_graph_lin_reg_int_3D_no_conf
                request.session['reg_graph_lin_reg_int_3D_no_conf'] = reg_graph_lin_reg_int_3D_no_conf
            
            residuals_plots_int = []
            for idx, header in enumerate(headers_lin_reg_int[1:], start=1):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=X.iloc[:, idx],
                    y=std_residuals,
                    mode='markers',
                    marker=dict(color="blue", opacity=0.7),
                    name='Residuals'
                ))

                fig.add_trace(go.Scatter(
                    x=[min(X.iloc[:, idx]), max(X.iloc[:, idx])],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color="black", dash="dash"),
                    name='Zero Line'
                ))

                fig.add_trace(go.Scatter(
                    x=[min(X.iloc[:, idx]), max(X.iloc[:, idx])],
                    y=[3, 3],
                    mode='lines',
                    line=dict(color="red", dash="dash"),
                    name='Upper Bound (+3)'
                ))

                fig.add_trace(go.Scatter(
                    x=[min(X.iloc[:, idx]), max(X.iloc[:, idx])],
                    y=[-3, -3],
                    mode='lines',
                    line=dict(color="red", dash="dash"),
                    name='Lower Bound (-3)'
                ))

                fig.update_layout(
                    title=f"Residuals vs {header}",
                    xaxis_title=header,
                    yaxis_title="Standardized Residuals",
                    showlegend=False,
                    template="plotly_white"
                )

                residual_plot_html = fig.to_html(full_html=False)
                residuals_plots_int.append(residual_plot_html)
            context['residuals_plots_int'] = residuals_plots_int
            request.session['residuals_plots_int'] = residuals_plots_int

            outliers_int = []
            for idx, residual in enumerate(std_residuals):
                if residual > 3 or residual < -3:
                    outlier_info = {
                        'index': idx+1,
                        'X_values': X.iloc[idx, 1:].tolist(),
                        'residual': round(residual, 4),
                        'actual_value': Y.iloc[idx],
                    }
                    outliers_int.append(outlier_info)
            context['outliers_int'] = outliers_int
            request.session['outliers_int'] = outliers_int

        except ValueError as e:
            context['error_lin_reg_int'] = str(e)
            context['results_lin_reg_int'] = None
            context['reg_graph_lin_reg_int_3D'] = None
            context['reg_graph_lin_reg_int_3D_no_conf'] = None
            context['model_description_lin_reg_int'] = None
            context['residuals_plots_int'] = None
            context['outliers_int'] = None
            return render(request, "regression/regression.html", context)

####################################################################################################
    if request.method == "POST" and tab == "nonlin_reg":
        data_nonlin_reg = request.POST.get('data_nonlin_reg')
        use_first_row_as_header_nonlin_reg = request.POST.get('use_first_row_as_header_nonlin_reg') == 'on'
        nonlinear_function = request.POST.get('nonlinear_function')

        context['use_first_row_as_header_nonlin_reg'] = 'checked' if use_first_row_as_header_nonlin_reg else ''
        request.session['use_first_row_as_header_nonlin_reg'] = use_first_row_as_header_nonlin_reg
        context['data_nonlin_reg'] = data_nonlin_reg
        request.session['data_nonlin_reg'] = data_nonlin_reg
        context['nonlinear_function'] = nonlinear_function
        request.session['nonlinear_function'] = nonlinear_function

        if not data_nonlin_reg.strip():
            context['error_nonlin_reg'] = "Please enter data before calculating."
            context['results_nonlin_reg'] = None
            context['nonlin_graph'] = None
            context['nonlin_graph_no_conf'] = None
            return render(request, "regression/regression.html", context)
        
        data_nonlin_reg = data_nonlin_reg.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_nonlin_reg.split('\n')]

        headers_nonlin_reg = []
        if use_first_row_as_header_nonlin_reg and rows:
            headers_nonlin_reg = rows[0]
            rows = rows[1:]

        if not headers_nonlin_reg:
            headers_nonlin_reg = ["Y", "X"]

        context['headers_nonlin_reg'] = headers_nonlin_reg
        request.session['headers_nonlin_reg'] = headers_nonlin_reg

        try:
            data = np.array([[float(value.strip()) for value in row] for row in rows])
            X, Y = data[:, 1], data[:, 0]

            def cuadratic(x, a, b, c):
                return a * x**2 + b * x + c
            
            def cubic(x, a, b, c, d):
                return a * x**3 + b * x**2 + c * x + d

            def exponential(x, a, b):
                return a * np.exp(b * x)

            def logarithmic(x, a, b):
                return a + b * np.log(x)
            
            def power(x, a, b):
                return a * np.power(x, b)

            def sinusoidal(x, a, b, c):
                return a * np.sin(b * x + c)

            def gaussian(x, a, b, c):
                return a * np.exp(-np.power(x - b, 2) / (2 * c**2))

            def hyperbolic(x, a, b):
                return a + b / x

            def sigmoid(x, a, b, c):
                return a / (1 + np.exp(-b * (x - c)))
            
            def logistic(x, a, b, c):
                return a / (1 + np.exp(-b * (x - c)))

            if nonlinear_function == "cuadratic":
                param_names = ['a', 'b', 'c']
                func = cuadratic
                initial_params = [1, 1, 1]
            elif nonlinear_function == "cubic":
                param_names = ['a', 'b', 'c', 'd']
                func = cubic
                initial_params = [1, 1, 1, 1]
            elif nonlinear_function == "exponential":
                param_names = ['a', 'b']
                func = exponential
                initial_params = [1, 0.1]
            elif nonlinear_function == "logarithmic":
                param_names = ['a', 'b']
                func = logarithmic
                initial_params = [1, 0.1]
            elif nonlinear_function == "power":
                param_names = ['a', 'b']
                func = power
                initial_params = [1, 1]
            elif nonlinear_function == "sinusoidal":
                param_names = ['a', 'b', 'c']
                func = sinusoidal
                initial_params = [1, 1, 0]
            elif nonlinear_function == "gaussian":
                param_names = ['a', 'b', 'c']
                func = gaussian
                initial_params = [1, 0, 1]
            elif nonlinear_function == "hyperbolic":
                param_names = ['a', 'b']
                func = hyperbolic
                initial_params = [1, 1]
            elif nonlinear_function == "sigmoid":
                param_names = ['a', 'b', 'c']
                func = sigmoid
                initial_params = [1, 1, 0.5]
            elif nonlinear_function == "logistic":
                param_names = ['a', 'b', 'c']
                func = logistic
                initial_params = [1, 1, np.median(X)]
            else:
                raise ValueError("Invalid function type.")

            params, pcov = curve_fit(func, X, Y, p0=initial_params)

            X_dense = np.linspace(min(X), max(X), 1000)
            fitted_values_dense = func(X_dense, *params)

            perr = np.sqrt(np.diag(pcov))

            conf_interval_lower = params - 1.96 * perr
            conf_interval_upper = params + 1.96 * perr

            residuals = Y - func(X, *params)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((Y - np.mean(Y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            equation = generate_equation(nonlinear_function, params)

            y_conf_lower = func(X_dense, *(params - 1.96 * perr))
            y_conf_upper = func(X_dense, *(params + 1.96 * perr))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=Y, mode='markers'))
            fig.add_trace(go.Scatter(x=X_dense, y=fitted_values_dense, mode='lines'))

            fig.update_layout(
                title="Nonlinear Regression",
                xaxis_title=headers_nonlin_reg[1],
                yaxis_title=headers_nonlin_reg[0],
                showlegend=False,
                template="plotly_white"
            )

            context['nonlin_graph_no_conf'] = fig.to_html(full_html=False)
            request.session['nonlin_graph_no_conf'] = context['nonlin_graph_no_conf']

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=Y, mode='markers'))
            fig.add_trace(go.Scatter(x=X_dense, y=fitted_values_dense, mode='lines', name='Adjusted Model', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=X_dense, y=y_conf_lower, fill=None, mode='lines', line=dict(color='lightgray')))
            fig.add_trace(go.Scatter(x=X_dense, y=y_conf_upper, fill='tonexty', mode='lines', line=dict(color='lightgray')))

            fig.update_layout(
                title="Nonlinear Regression (95% Confidence Band)",
                xaxis_title=headers_nonlin_reg[1],
                yaxis_title=headers_nonlin_reg[0],
                showlegend=False,
                template="plotly_white"
            )

            context['nonlin_graph'] = fig.to_html(full_html=False)
            request.session['nonlin_graph'] = context['nonlin_graph']

            results_nonlin_reg = {
                "R_squared": r_squared,
                "ss_res": ss_res,
                "ss_tot": ss_tot,
                "params": [],
                "equation": equation
            }

            for i, param_name in enumerate(param_names):
                results_nonlin_reg["params"].append((
                    param_name,
                    params[i],
                    perr[i],
                    params[i] / perr[i],  # t-value
                    2 * (1 - stats.t.cdf(np.abs(params[i] / perr[i]), df=len(X) - len(params))),  # p-value
                    conf_interval_lower[i],
                    conf_interval_upper[i]
                ))

            context['results_nonlin_reg'] = results_nonlin_reg
            request.session['results_nonlin_reg'] = context['results_nonlin_reg']

        except Exception as e:
            context['error_nonlin_reg'] = str(e)
            context['results_nonlin_reg'] = None
            context['nonlin_graph'] = None
            context['nonlin_graph_no_conf'] = None

    return render(request, "regression/regression.html", context)


def generate_equation(func_name, params):
    if func_name == "cuadratic":
        return f"y = {params[0]:.4f}x^2 + {params[1]:.4f}x + {params[2]:.4f}"
    elif func_name == "cubic":
        return f"y = {params[0]:.4f}x^3 + {params[1]:.4f}x^2 + {params[2]:.4f}x + {params[3]:.4f}"
    elif func_name == "exponential":
        return f"y = {params[0]:.4f}e^({params[1]:.4f}x)"
    elif func_name == "logarithmic":
        return f"y = {params[0]:.4f} + {params[1]:.4f}ln(x)"
    elif func_name == "power":
        return f"y = {params[0]:.4f}x^{params[1]:.4f}"
    elif func_name == "sinusoidal":
        return f"y = {params[0]:.4f}sin({params[1]:.4f}x + {params[2]:.4f})"
    elif func_name == "gaussian":
        return f"y = {params[0]:.4f}e^(-(x - {params[1]:.4f})^2 / (2 * {params[2]:.4f}^2))"
    elif func_name == "hyperbolic":
        return f"y = {params[0]:.4f} + {params[1]:.4f}/x"
    elif func_name == "sigmoid":
        return f"y = {params[0]:.4f}/(1 + e^(-{params[1]:.4f}(x - {params[2]:.4f})))"
    elif func_name == "logistic":
        return f"y = {params[0]:.4f}/(1 + e^(-{params[1]:.4f}(x - {params[2]:.4f})))"
    else:
        return "Unknown function"