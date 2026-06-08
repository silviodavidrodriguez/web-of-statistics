from django.shortcuts import render
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from collections import Counter
from sklearn.model_selection import train_test_split
import re

def normalize_data(data, method):
    if method == "none":
        return data

    scalers = {
        "zscore": StandardScaler(),
        "minmax": MinMaxScaler(),
        "maxabs": MaxAbsScaler(),
        "robust": RobustScaler(),
    }

    if method not in scalers:
        raise ValueError("Invalid normalization method selected.")

    return scalers[method].fit_transform(data)

def select_simca_components_loo(X_class):
    n_samples, n_variables = X_class.shape
    max_components = min(n_samples - 1, n_variables)

    if max_components < 1:
        return 1, []

    loo_errors = []

    for n_comp in range(1, max_components + 1):
        errors = []

        for i in range(n_samples):
            X_train = np.delete(X_class, i, axis=0)
            X_test = X_class[i:i + 1]

            pca = PCA(n_components=n_comp)
            pca.fit(X_train)

            scores_test = pca.transform(X_test)
            X_reconstructed = pca.inverse_transform(scores_test)

            error = np.sum((X_test - X_reconstructed) ** 2)
            errors.append(error)

        mean_error = np.mean(errors)
        loo_errors.append({
            "components": n_comp,
            "mean_reconstruction_error": mean_error,
        })

    min_error = min(row["mean_reconstruction_error"] for row in loo_errors)
    error_range = max(row["mean_reconstruction_error"] for row in loo_errors) - min_error
    tolerance = min_error + (0.05 * error_range)

    eligible = [
        row for row in loo_errors
        if row["mean_reconstruction_error"] <= tolerance
    ]

    best = min(eligible, key=lambda row: row["components"])

    return best["components"], loo_errors

def calculate_simca_diagnostics(X, pca_model):
    scores = pca_model.transform(X)
    X_reconstructed = pca_model.inverse_transform(scores)
    residuals = X - X_reconstructed

    q_residuals = np.sum(residuals ** 2, axis=1)

    eigenvalues = pca_model.explained_variance_
    eigenvalues = np.where(eigenvalues == 0, 1e-12, eigenvalues)

    hotelling_t2 = np.sum((scores ** 2) / eigenvalues, axis=1)

    return hotelling_t2, q_residuals

def evaluate_simca_sample(sample, simca_models):
    accepted_models = []

    for model_class, model_info in simca_models.items():
        pca_model = model_info["pca_model"]
        t2_limit = model_info["t2_limit"]
        q_limit = model_info["q_limit"]

        t2_sample, q_sample = calculate_simca_diagnostics(sample, pca_model)

        accepted = (
            t2_sample[0] <= t2_limit
            and q_sample[0] <= q_limit
        )

        if accepted:
            accepted_models.append(str(model_class))

    if len(accepted_models) == 1:
        predicted_class = accepted_models[0]
    elif len(accepted_models) == 0:
        predicted_class = "Rejected"
    else:
        predicted_class = "Multiple"

    return accepted_models, predicted_class

def multivariate(request):
    tab = request.GET.get('tab', None)
    subtab = request.GET.get('subtab', None)

    context = {
        'segment': 'multivariate',
        'active_tab': tab,
        'active_subtab': subtab,
        'data_pca': request.session.get('data_pca', ""),
        'results_pca': request.session.get('results_pca', None),
        'explained_variance': request.session.get('explained_variance', None),
        'loadings': request.session.get('loadings', None),
        'components': request.session.get('components', None),
        'component_names': request.session.get('component_names', None),
        'components_2D_graph': request.session.get('components_2D_graph', None),
        'components_3D_graph': request.session.get('components_3D_graph', None),
        'loadings_graph': request.session.get('loadings_graph', None),
        'scree_plot': request.session.get('scree_plot', None),
        'headers_pca': request.session.get('headers_pca', None),
        'use_first_row_as_header_pca': 'checked' if request.session.get('use_first_row_as_header_pca', False) else '',
        'use_first_column_as_id_pca': 'checked' if request.session.get('use_first_column_as_id_pca', False) else '',
        'use_last_column_as_factor_pca': 'checked' if request.session.get('use_last_column_as_factor_pca', False) else '',
        'data_h_clus': request.session.get('data_h_clus', ""),
        'h_clus_graph': request.session.get('h_clus_graph', None),
        'use_first_column_as_id_h_clus': 'checked' if request.session.get('use_first_column_as_id_h_clus', False) else '',
        'data_k_means': request.session.get('data_k_means', ""),
        'silhouette_avg': request.session.get('silhouette_avg', None),
        'num_clusters': request.session.get('num_clusters', None),
        'k_means_graph': request.session.get('k_means_graph', None),
        'use_first_column_as_id_k_means': 'checked' if request.session.get('use_first_column_as_id_k_means', False) else '',
        'data_correlation': request.session.get('data_correlation', ""),
        'correlation_graph': request.session.get('correlation_graph', None),
        'headers_correlation': request.session.get('headers_correlation', None),
        'use_first_row_as_header_correlation': 'checked' if request.session.get('use_first_row_as_header_correlation', False) else '',
        'normalization_pca': request.session.get('normalization_pca', 'none'),
        'normalization_h_clus': request.session.get('normalization_h_clus', 'none'),
        'normalization_k_means': request.session.get('normalization_k_means', 'none'),
        'data_simca': request.session.get('data_simca', ""),
        'use_first_row_as_header_simca': 'checked' if request.session.get('use_first_row_as_header_simca', False) else '',
        'use_first_column_as_id_simca': 'checked' if request.session.get('use_first_column_as_id_simca', False) else '',
        'normalization_simca': request.session.get('normalization_simca', 'zscore'),
        'results_simca': request.session.get('results_simca', None),
        'simca_pc_selection_method': request.session.get('simca_pc_selection_method', 'variance'),
        'simca_variance_threshold': request.session.get('simca_variance_threshold', 95),
        'simca_n_components': request.session.get('simca_n_components', 2),
        'simca_confidence_level':request.session.get('simca_confidence_level',95),
        'simca_coomans_class_x': request.session.get('simca_coomans_class_x', ''),
        'simca_coomans_class_y': request.session.get('simca_coomans_class_y', ''),
        'simca_validation_mode': request.session.get('simca_validation_mode','training'),
        'simca_validation_limit_strategy':request.session.get('simca_validation_limit_strategy','percentile'),
        'data_simca_external': request.session.get('data_simca_external', ""),
        'external_has_class_simca': 'checked' if request.session.get('external_has_class_simca', False) else '',
        }

    if request.method == "POST" and request.POST.get("clear_pca") == "true":
        if 'data_pca' in request.session:
            del request.session['data_pca']
        if 'results_pca' in request.session:
            del request.session['results_pca']
        if 'explained_variance' in request.session:
            del request.session['explained_variance']
        if 'loadings' in request.session:
            del request.session['loadings']
        if 'components' in request.session:
            del request.session['components']
        if 'components_2D_graph' in request.session:
            del request.session['components_2D_graph']
        if 'components_3D_graph' in request.session:
            del request.session['components_3D_graph']
        if 'loadings_graph' in request.session:
            del request.session['loadings_graph']
        if 'scree_plot' in request.session:
            del request.session['scree_plot']
        if 'headers_pca' in request.session:
            del request.session['headers_pca']
        if 'use_first_row_as_header_pca' in request.session:
            request.session.pop('use_first_row_as_header_pca', False)
        if 'use_first_column_as_id_pca' in request.session:
            request.session.pop('use_first_column_as_id_pca', False)
        if 'use_last_column_as_factor_pca' in request.session:
            request.session.pop('use_last_column_as_factor_pca', False)
        context['data_pca'] = ""
        context['results_pca'] = None
        context['explained_variance'] = None
        context['loadings'] = None
        context['components'] = None
        context['components_2D_graph'] = None
        context['components_3D_graph'] = None
        context['loadings_graph'] = None
        context['scree_plot'] = None
        context['headers_pca'] = None
        context['use_first_row_as_header_pca'] = False
        context['use_first_column_as_id_pca'] = False
        context['use_last_column_as_factor_pca'] = False
        return render(request, "multivariate/multivariate.html", context)
    
    if request.method == "POST" and request.POST.get("clear_h_clus") == "true":
        if 'data_h_clus' in request.session:
            del request.session['data_h_clus']
        if 'h_clus_graph' in request.session:
            del request.session['h_clus_graph']
        if 'uuse_first_column_as_id_h_clus' in request.session:
            request.session.pop('use_first_column_as_id_h_clus', False)
        context['data_h_clus'] = ""
        context['h_clus_graph'] = None
        context['use_first_column_as_id_h_clus'] = False
        return render(request, "multivariate/multivariate.html", context)

    if request.method == "POST" and request.POST.get("clear_k_means") == "true":
        if 'data_k_means' in request.session:
            del request.session['data_k_means']
        if 'silhouette_avg' in request.session:
            del request.session['silhouette_avg']
        if 'num_clusters' in request.session:
            del request.session['num_clusters']
        if 'k_means_graph' in request.session:
            del request.session['k_means_graph']
        if 'use_first_column_as_id_k_means' in request.session:
            request.session.pop('use_first_column_as_id_k_means', False)
        context['data_k_means'] = ""
        context['silhouette_avg'] = None
        context['num_clusters'] = None
        context['k_means_graph'] = None
        context['use_first_column_as_id_k_means'] = False
        return render(request, "multivariate/multivariate.html", context)
    
    if request.method == "POST" and request.POST.get("clear_correlation") == "true":
        if 'data_correlation' in request.session:
            del request.session['data_correlation']
        if 'correlation_graph' in request.session:
            del request.session['correlation_graph']
        if 'headers_correlation' in request.session:
            del request.session['headers_correlation']
        if 'use_first_row_as_header_correlation' in request.session:
            request.session.pop('use_first_row_as_header_correlation', False)
        context['data_correlation'] = ""
        context['correlation_graph'] = None
        context['headers_correlation'] = None
        context['use_first_row_as_header_correlation'] = False
        return render(request, "multivariate/multivariate.html", context)
    
    if request.method == "POST" and request.POST.get("clear_simca") == "true":
        for key in [
            'data_simca',
            'use_first_row_as_header_simca',
            'use_first_column_as_id_simca',
            'normalization_simca',
            'results_simca',
            'simca_pc_selection_method',
            'simca_n_components',
            'simca_variance_threshold',
            'simca_confidence_level',
            'simca_coomans_class_x',
            'simca_coomans_class_y',
            'simca_validation_mode',
            'simca_validation_limit_strategy',
            'data_simca_external',
            'external_has_class_simca',
        ]:
            request.session.pop(key, None)

        context['data_simca'] = ""
        context['use_first_row_as_header_simca'] = ""
        context['use_first_column_as_id_simca'] = ""
        context['normalization_simca'] = "zscore"
        context['results_simca'] = None
        context['simca_n_components'] = 2
        context['simca_pc_selection_method'] = "variance"
        context['simca_validation_mode'] = "training"
        context['simca_validation_limit_strategy'] = "percentile"
        context['simca_variance_threshold'] = 95
        context['simca_confidence_level'] = 95
        context['simca_coomans_class_x'] = ""
        context['simca_coomans_class_y'] = ""
        context['data_simca_external'] = ""
        context['external_has_class_simca'] = ""

        return render(request, "multivariate/multivariate.html", context)

#########################################################################################
    if request.method == "POST" and tab == "pca":
        data_pca = request.POST.get('data_pca')
        use_first_row_as_header_pca = request.POST.get('use_first_row_as_header_pca') == 'on'
        use_first_column_as_id_pca = request.POST.get('use_first_column_as_id_pca') == 'on'
        use_last_column_as_factor_pca = request.POST.get('use_last_column_as_factor_pca') == 'on'
        normalization_pca = request.POST.get('normalization_pca', 'none')

        context['normalization_pca'] = normalization_pca
        request.session['normalization_pca'] = normalization_pca
        context['use_first_row_as_header_pca'] = 'checked' if use_first_row_as_header_pca else ''
        request.session['use_first_row_as_header_pca'] = use_first_row_as_header_pca
        context['use_first_column_as_id_pca'] = 'checked' if use_first_column_as_id_pca else ''
        request.session['use_first_column_as_id_pca'] = use_first_column_as_id_pca
        context['use_last_column_as_factor_pca'] = 'checked' if use_last_column_as_factor_pca else ''
        request.session['use_last_column_as_factor_pca'] = use_last_column_as_factor_pca
        context['data_pca'] = data_pca
        request.session['data_pca'] = data_pca

        if not data_pca.strip():
            context['error_pca'] = "Please enter data before calculating."
            context['results_pca'] = None
            context['explained_variance'] = None
            context['loadings'] = None
            context['components'] = None
            context['components_2D_graph'] = None
            context['components_3D_graph'] = None
            context['scree_plot'] = None
            context['loadings_graph'] = None
            return render(request, "multivariate/multivariate.html", context)
        
        data_pca = data_pca.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_pca.split('\n')]
        columns = []
        row_lengths = []
        headers_pca = []
        id_pca = []
        factor_pca = []

        if use_first_row_as_header_pca and rows:
            headers_pca = rows[0]
            rows = rows[1:]
        
        if headers_pca:
            expected_with_id_and_factor = len(rows[0]) if rows else len(headers_pca)

            if use_first_column_as_id_pca and len(headers_pca) == expected_with_id_and_factor:
                headers_pca = headers_pca[1:]

            if use_last_column_as_factor_pca and len(headers_pca) > 0:
                numeric_cols_after_options = len(rows[0])
                if use_first_column_as_id_pca:
                    numeric_cols_after_options -= 1
                if use_last_column_as_factor_pca:
                    numeric_cols_after_options -= 1

                if len(headers_pca) > numeric_cols_after_options:
                    headers_pca = headers_pca[:-1]

        if use_first_column_as_id_pca and rows:
            id_pca = [row[0] for row in rows]
            rows = [row[1:] for row in rows]
        else:
            id_pca = [str(i + 1) for i in range(len(rows))]

        if use_last_column_as_factor_pca and rows:
            factor_pca = [row[-1] for row in rows]
            rows = [row[:-1] for row in rows]

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
                            if row_idx == 0 and not use_first_row_as_header_pca:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)
                row_lengths.append(len(float_row))

            if len(set(row_lengths)) > 1:
                raise ValueError("Missing value/s detected. Please ensure all values are present.")

            data_columns = np.array(columns).T
            data_columns = data_columns.T
            data_columns = normalize_data(data_columns, normalization_pca)

            if not headers_pca:
                headers_pca = [f"var {i + 1}" for i in range(len(data_columns[0]))]
            
            context['headers_pca'] = headers_pca
            request.session['headers_pca'] = headers_pca
            context['id_pca'] = id_pca
            request.session['id_pca'] = id_pca
            context['factor_pca'] = factor_pca
            request.session['factor_pca'] = factor_pca

            if len(data_columns[0]) < 3:
                context['error_pca'] = "At least 3 columns are required for PCA."
                context['results_pca'] = None
                context['components_2D_graph'] = None
                context['components_3D_graph'] = None
                context['loadings_graph'] = None
                return render(request, "multivariate/multivariate.html", context)
            else:
                pca = PCA()
                components = pca.fit_transform(data_columns)
                explained_variance = pca.explained_variance_ratio_
                loadings = pca.components_.T
                colors = factor_pca if factor_pca else ["none"] * len(id_pca)

                fig_2d = px.scatter(
                    x=components[:, 0],
                    y=components[:, 1],
                    color=colors,
                    hover_name=id_pca,
                    title="2D Scores Plot",
                    labels={"x": f"PC1 ({explained_variance[0]*100:.2f}%)", "y": f"PC2 ({explained_variance[1]*100:.2f}%)"}
                )
                fig_2d.update_layout(
                    title_x=0.5
                )

                fig_3d = px.scatter_3d(
                    x=components[:, 0],
                    y=components[:, 1],
                    z=components[:, 2],
                    color=colors,
                    hover_name=id_pca,
                    title=f"3D Scores Plot",
                    labels={"x": f"PC1 ({explained_variance[0]*100:.2f}%)",
                            "y": f"PC2 ({explained_variance[1]*100:.2f}%)",
                            "z": f"PC3 ({explained_variance[2]*100:.2f}%)"}
                )
                fig_3d.update_layout(
                    title_x=0.5,
                    height=750,
                    margin=dict(l=20, r=20, t=70, b=80),
                    scene=dict(
                        xaxis_title=f"PC1 ({explained_variance[0]*100:.2f}%)",
                        yaxis_title=f"PC2 ({explained_variance[1]*100:.2f}%)",
                        zaxis_title=f"PC3 ({explained_variance[2]*100:.2f}%)",
                        aspectmode="cube"
                    )
                )
                fig_3d.update_traces(marker=dict(size=6))

                cumulative_variance = np.cumsum(explained_variance)
                fig_scree = go.Figure()
                fig_scree.add_trace(go.Bar(
                    x=[f"PC{i+1}" for i in range(len(explained_variance))],
                    y=explained_variance * 100,
                    name="Exp. Var. (%)"
                ))
                fig_scree.add_trace(go.Scatter(
                    x=[f"PC{i+1}" for i in range(len(explained_variance))],
                    y=cumulative_variance * 100,
                    mode="lines+markers",
                    name="Cum. Var. (%)"
                ))
                fig_scree.update_layout(
                    title="Scree Plot",
                    xaxis_title="Principal Components",
                    yaxis_title="Variance (%)",
                    title_x=0.5
                )

                loadings_fig = go.Figure()
                for i, component in enumerate(loadings.T):
                    sorted_indices = np.argsort(-np.abs(component))
                    sorted_variables = [headers_pca[idx] for idx in sorted_indices]
                    sorted_values = component[sorted_indices]

                    loadings_fig.add_trace(go.Bar(
                        x=sorted_variables,
                        y=sorted_values,
                        name=f"PC{i+1}",
                        visible=(i == 0)
                    ))

                buttons = [
                    {
                        "label": f"PC{i+1}",
                        "method": "update",
                        "args": [
                            {"visible": [j == i for j in range(len(loadings.T))]},
                            {"title": f"Loadings for PC{i+1}"}
                        ]
                    }
                    for i in range(len(loadings.T))
                ]

                loadings_fig.update_layout(
                    title="Loadings for PC1",
                    xaxis_title="Variables",
                    yaxis_title="Loadings",
                    title_x=0.5,
                    updatemenus=[{
                        "buttons": buttons,
                        "direction": "down",
                        "showactive": True
                    }]
                )

                results_pca = {
                    "explained_variance": [
                        {
                            "component": f"PC{i + 1}",
                            "eigenvalue": f"{pca.explained_variance_[i]:.5f}",
                            "variance_ratio": f"{explained_variance[i] * 100:.2f}",
                            "cumulative_variance": f"{cumulative_variance[i] * 100:.2f}",
                        }
                        for i in range(len(explained_variance))
                    ],
                    "loadings": [
                        {
                            "variable": headers_pca[i],
                            "values": [f"{val:.5f}" for val in loadings[i]]
                        }
                        for i in range(len(headers_pca))
                    ],
                    "components": [
                        {
                            "sample_id": id_pca[i],
                            "group": factor_pca[i] if factor_pca else "",
                            "values": [f"{val:.5f}" for val in components[i]]
                        }
                        for i in range(len(id_pca))
                    ],
                    "component_names": [f"PC{i + 1}" for i in range(len(explained_variance))]
                }

                context['results_pca'] = results_pca
                request.session['results_pca'] = results_pca
                context['explained_variance'] = results_pca['explained_variance']
                request.session['explained_variance'] = context['explained_variance']
                context['loadings'] = results_pca['loadings']
                request.session['loadings'] = context['loadings']
                context['components'] = results_pca['components']
                request.session['components'] = context['components']
                context['component_names'] = results_pca['component_names']
                request.session['component_names'] = context['component_names']
                context['components_2D_graph'] = fig_2d.to_html(full_html=False)
                context['components_2D_graph'] = context['components_2D_graph']
                request.session['components_2D_graph'] = context['components_2D_graph']
                context['components_3D_graph'] = fig_3d.to_html(full_html=False)
                context['components_3D_graph'] = context['components_3D_graph']
                request.session['components_3D_graph'] = context['components_3D_graph']
                context['scree_plot'] = fig_scree.to_html(full_html=False)
                context['scree_plot'] = context['scree_plot']
                request.session['scree_plot'] = context['scree_plot']
                context['loadings_graph'] = loadings_fig.to_html(full_html=False)
                context['loadings_graph'] = context['loadings_graph']
                request.session['loadings_graph'] = context['loadings_graph']

        except ValueError as e:
            context['error_pca'] = str(e)
            context['results_pca'] = None
            context['explained_variance'] = None
            context['loadings'] = None
            context['components'] = None
            context['components_2D_graph'] = None
            context['components_3D_graph'] = None
            context['scree_plot'] = None
            context['loadings_graph'] = None

#########################################################################################
    if request.method == "POST" and tab == "cluster" and subtab == "h_clus":
        data_h_clus = request.POST.get('data_h_clus')
        use_first_column_as_id_h_clus = request.POST.get('use_first_column_as_id_h_clus') == 'on'
        normalization_h_clus = request.POST.get('normalization_h_clus', 'none')

        context['normalization_h_clus'] = normalization_h_clus
        request.session['normalization_h_clus'] = normalization_h_clus
        context['use_first_column_as_id_h_clus'] = 'checked' if use_first_column_as_id_h_clus else ''
        request.session['use_first_column_as_id_h_clus'] = use_first_column_as_id_h_clus
        context['data_h_clus'] = data_h_clus
        request.session['data_h_clus'] = data_h_clus

        if not data_h_clus.strip():
            context['error_h_clus'] = "Please enter data before calculating."
            context['h_clus_graph'] = None
            return render(request, "multivariate/multivariate.html", context)
        
        data_h_clus = data_h_clus.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_h_clus.split('\n')]
        columns = []
        row_lengths = []
        id_h_clus = []

        if use_first_column_as_id_h_clus and rows:
                id_h_clus = [row[0] for row in rows]
                rows = [row[1:] for row in rows]
        else:
            id_h_clus = [str(i + 1) for i in range(len(rows))]

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
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)
                row_lengths.append(len(float_row))

            if len(set(row_lengths)) > 1:
                raise ValueError("Missing value/s detected. Please ensure all values are present.")

            data_columns = np.array(columns).T
            data_columns = data_columns.T
            data_columns = normalize_data(data_columns, normalization_h_clus)

            fig = ff.create_dendrogram(data_columns, labels=id_h_clus, linkagefun=lambda x: sch.linkage(x, 'ward'))
            fig.update_layout(
                title="Hierarchical Cluster Dendrogram (Ward)",
                title_x=0.5,
                showlegend=False
            )
            context['h_clus_graph'] = fig.to_html(full_html=False)
            request.session['h_clus_graph'] = context['h_clus_graph']

        except ValueError as e:
            context['error_h_clus'] = str(e)
            context['h_clus_graph'] = None

#########################################################################################
    if request.method == "POST" and tab == "cluster" and subtab == "k_means":
        data_k_means = request.POST.get('data_k_means')
        use_first_column_as_id_k_means = request.POST.get('use_first_column_as_id_k_means') == 'on'
        num_clusters = request.POST.get('num_clusters')
        normalization_k_means = request.POST.get('normalization_k_means', 'none')

        context['normalization_k_means'] = normalization_k_means
        request.session['normalization_k_means'] = normalization_k_means
        context['use_first_column_as_id_k_means'] = 'checked' if use_first_column_as_id_k_means else ''
        request.session['use_first_column_as_id_k_means'] = use_first_column_as_id_k_means
        context['data_k_means'] = data_k_means
        request.session['data_k_means'] = data_k_means
        context['num_clusters'] = num_clusters
        request.session['num_clusters'] = num_clusters 

        if not data_k_means.strip():
            context['error_k_means'] = "Please enter data before calculating."
            context['k_means_graph'] = None
            context['silhouette_avg'] = None
            context['num_clusters'] = None
            return render(request, "multivariate/multivariate.html", context)

        if not num_clusters.isdigit():
            context['error_k_means'] = "Number of clusters must be an integer."
            context['k_means_graph'] = None
            context['silhouette_avg'] = None
            context['num_clusters'] = None
            return render(request, "multivariate/multivariate.html", context)

        num_clusters = int(num_clusters)
        if num_clusters <= 1:
            context['error_k_means'] = "Number of clusters must be greater than 1."
            context['k_means_graph'] = None
            context['silhouette_avg'] = None
            context['num_clusters'] = None
            return render(request, "multivariate/multivariate.html", context)

        data_k_means = data_k_means.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_k_means.split('\n')]
        columns = []
        row_lengths = []
        id_k_means = []

        if use_first_column_as_id_k_means and rows:
            id_k_means = [row[0] for row in rows]
            rows = [row[1:] for row in rows]
        else:
            id_k_means = [str(i + 1) for i in range(len(rows))]

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
                            raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)
                row_lengths.append(len(float_row))

            if len(set(row_lengths)) > 1:
                raise ValueError("Missing value/s detected. Please ensure all values are present.")

            data_columns = np.array(columns).T
            data_columns = data_columns.T
            data_columns = normalize_data(data_columns, normalization_k_means)

            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_assignments = kmeans.fit_predict(data_columns)
            silhouette_avg = silhouette_score(data_columns, cluster_assignments)
            sample_silhouette_values = silhouette_samples(data_columns, cluster_assignments)

            fig = go.Figure()
            y_lower = 10

            for i in range(num_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_assignments == i]
                ith_cluster_ids = [id_k_means[j] for j in np.where(cluster_assignments == i)[0]]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                fig.add_trace(go.Bar(
                    x=ith_cluster_silhouette_values,
                    y=ith_cluster_ids,  # Usar los IDs como etiquetas del eje Y
                    orientation='h',
                    name=f'Cluster {i}',
                    marker=dict(opacity=0.7)
                ))

                y_lower = y_upper + 10

            fig.add_vline(x=silhouette_avg, line=dict(color="red", dash="dash"), name="Average Silhouette")

            fig.update_layout(
                title="Silhouette Plot for K-means Clustering",
                xaxis_title="Silhouette coefficient values",
                yaxis_title="Sample ID",
                showlegend=False,
                height=600
            )

            context['k_means_graph'] = fig.to_html(full_html=False)
            request.session['k_means_graph'] = context['k_means_graph']
            context['silhouette_avg'] = round(silhouette_avg, 5)
            request.session['silhouette_avg'] = context['silhouette_avg']

        except ValueError as e:
            context['error_k_means'] = str(e)
            context['k_means_graph'] = None
            context['silhouette_avg'] = None
            context['num_clusters'] = None

#########################################################################################
    if request.method == "POST" and tab == "correlation":
        data_correlation = request.POST.get('data_correlation')
        use_first_row_as_header_correlation = request.POST.get('use_first_row_as_header_correlation') == 'on'

        context['use_first_row_as_header_correlation'] = 'checked' if use_first_row_as_header_correlation else ''
        request.session['use_first_row_as_header_correlation'] = use_first_row_as_header_correlation
        context['data_correlation'] = data_correlation
        request.session['data_correlation'] = data_correlation

        if not data_correlation.strip():
            context['error_correlation'] = "Please enter data before calculating."
            context['correlation_graph'] = None
            return render(request, "multivariate/multivariate.html", context)
        
        data_correlation = data_correlation.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_correlation.split('\n')]
        columns = []
        row_lengths = []
        headers_correlation = []

        if use_first_row_as_header_correlation and rows:
            headers_correlation = rows[0]
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
                            if row_idx == 0 and not use_first_row_as_header_correlation:
                                raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as header' is not checked.")
                            else:
                                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")
                columns.append(float_row)
                row_lengths.append(len(float_row))

            if len(set(row_lengths)) > 1:
                raise ValueError("Missing value/s detected. Please ensure all values are present.")

            data_columns = np.array(columns).T

            if not headers_correlation:
                headers_correlation = [f"var. {i + 1}" for i in range(len(data_columns))]
            
            context['headers_correlation'] = headers_correlation
            request.session['headers_correlation'] = headers_correlation

            if len(data_columns[0]) < 2:
                context['error_correlation'] = "At least 2 columns are required for PCA."
                context['correlation_graph'] = None
                return render(request, "multivariate/multivariate.html", context)
            else:
                correlation_matrix = np.corrcoef(data_columns)
                text_values = [[f"{value:.2f}" for value in row] for row in correlation_matrix]

                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    z=correlation_matrix,
                    x=headers_correlation,
                    y=headers_correlation,
                    colorscale='Viridis',
                    colorbar=dict(),
                    text=text_values,
                    texttemplate="%{text}",
                ))

                fig.update_layout(
                    height=600
                )

                context['correlation_graph'] = fig.to_html(full_html=False)
                request.session['correlation_graph'] = context['correlation_graph']

        except ValueError as e:
            context['error_correlation'] = str(e)
            context['correlation_graph'] = None

###########################################################################################
    if request.method == "POST" and tab == "simca":
        data_simca = request.POST.get('data_simca', '')
        use_first_row_as_header_simca = request.POST.get('use_first_row_as_header_simca') == 'on'
        use_first_column_as_id_simca = request.POST.get('use_first_column_as_id_simca') == 'on'
        normalization_simca = request.POST.get('normalization_simca', 'zscore')
        simca_pc_selection_method = request.POST.get('simca_pc_selection_method', 'variance')
        simca_validation_mode = request.POST.get('simca_validation_mode','training')
        simca_validation_limit_strategy = request.POST.get('simca_validation_limit_strategy','percentile')
        simca_n_components = request.POST.get('simca_n_components', 2)
        simca_variance_threshold = request.POST.get('simca_variance_threshold', 95)
        data_simca_external = request.POST.get('data_simca_external', '')
        external_has_class_simca = request.POST.get('external_has_class_simca') == 'on'
        
        try:
            simca_confidence_level = float(request.POST.get('simca_confidence_level',95))
        except ValueError:
            simca_confidence_level = 95.0
        simca_confidence_level = max(50.0,min(simca_confidence_level, 99.99))

        context['data_simca'] = data_simca
        request.session['data_simca'] = data_simca

        context['use_first_row_as_header_simca'] = 'checked' if use_first_row_as_header_simca else ''
        request.session['use_first_row_as_header_simca'] = use_first_row_as_header_simca

        context['use_first_column_as_id_simca'] = 'checked' if use_first_column_as_id_simca else ''
        request.session['use_first_column_as_id_simca'] = use_first_column_as_id_simca

        context['normalization_simca'] = normalization_simca
        request.session['normalization_simca'] = normalization_simca

        context['simca_confidence_level'] = simca_confidence_level
        request.session['simca_confidence_level'] = simca_confidence_level

        context['data_simca_external'] = data_simca_external
        request.session['data_simca_external'] = data_simca_external

        context['external_has_class_simca'] = 'checked' if external_has_class_simca else ''
        request.session['external_has_class_simca'] = external_has_class_simca

        try:
            simca_n_components = int(simca_n_components)
        except ValueError:
            simca_n_components = 2

        try:
            simca_variance_threshold = float(simca_variance_threshold)
        except ValueError:
            simca_variance_threshold = 95

        simca_variance_threshold = max(1, min(simca_variance_threshold, 100))

        context['simca_pc_selection_method'] = simca_pc_selection_method
        request.session['simca_pc_selection_method'] = simca_pc_selection_method

        context['simca_n_components'] = simca_n_components
        request.session['simca_n_components'] = simca_n_components

        context['simca_variance_threshold'] = simca_variance_threshold
        request.session['simca_variance_threshold'] = simca_variance_threshold

        context['simca_validation_mode'] = simca_validation_mode
        request.session['simca_validation_mode'] = simca_validation_mode

        context['simca_validation_limit_strategy'] = (simca_validation_limit_strategy)
        request.session['simca_validation_limit_strategy'] = simca_validation_limit_strategy

        if not data_simca.strip():
            context['error_simca'] = "Please enter training data before calculating."
            context['results_simca'] = None
            return render(request, "multivariate/multivariate.html", context)

        data_simca = data_simca.replace('\r', '').strip()
        rows = [row.split('\t') for row in data_simca.split('\n')]

        headers_simca = []
        id_simca = []
        class_simca = []
        columns = []
        row_lengths = []

        if use_first_row_as_header_simca and rows:
            headers_simca = rows[0]
            rows = rows[1:]

        if use_first_column_as_id_simca and rows:
            id_simca = [row[0] for row in rows]
            rows = [row[1:] for row in rows]
        else:
            id_simca = [str(i + 1) for i in range(len(rows))]

        if rows:
            class_simca = [row[-1] for row in rows]
            rows = [row[:-1] for row in rows]

        try:
            if not rows:
                raise ValueError("Please enter training data before calculating.")

            if not class_simca:
                raise ValueError("SIMCA requires the last column to contain class/group labels.")

            if len(set(class_simca)) < 2:
                raise ValueError("At least two classes/groups are required for SIMCA.")

            for row_idx, row in enumerate(rows):
                float_row = []

                for value in row:
                    value = value.strip()

                    if not value:
                        raise ValueError("Missing value/s detected. Please ensure all values are present.")

                    try:
                        float_row.append(float(value))
                    except ValueError:
                        if row_idx == 0 and not use_first_row_as_header_simca:
                            raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as variable names' is not checked.")
                        else:
                            raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")

                columns.append(float_row)
                row_lengths.append(len(float_row))

            if len(set(row_lengths)) > 1:
                raise ValueError("Missing value/s detected. Please ensure all rows have the same number of variables.")

            data_columns = np.array(columns)

            if data_columns.shape[1] < 2:
                raise ValueError("At least two numeric variables are required for SIMCA.")

            if headers_simca:
                if use_first_column_as_id_simca:
                    headers_simca = headers_simca[1:]

                if len(headers_simca) > data_columns.shape[1]:
                    headers_simca = headers_simca[:-1]

            if not headers_simca:
                headers_simca = [f"var {i + 1}" for i in range(data_columns.shape[1])]

            simca_scaler = None
            if normalization_simca == "none":
                data_columns_scaled = data_columns
            else:
                simca_scalers = {
                    "zscore": StandardScaler(),
                    "minmax": MinMaxScaler(),
                    "maxabs": MaxAbsScaler(),
                    "robust": RobustScaler(),
                }

                simca_scaler = simca_scalers[normalization_simca]
                data_columns_scaled = simca_scaler.fit_transform(data_columns)

            data_columns = data_columns_scaled

            class_counts = Counter(class_simca)

            class_simca_array = np.array(class_simca)
            simca_models_summary = []
            simca_diagnostics = []
            simca_models = {}
            simca_classification = []
            simca_external_classification = []
            simca_external_summary = None
            simca_external_confusion_matrix = None
            simca_external_sensitivity_specificity = []

            for cls in sorted(class_counts.keys()):
                X_class = data_columns[class_simca_array == cls]

                max_possible_components = min(X_class.shape[0], X_class.shape[1])

                if simca_pc_selection_method == "loo":
                    selected_components, loo_errors = select_simca_components_loo(X_class)
                    selected_components = int(selected_components)

                elif simca_pc_selection_method == "variance":
                    pca_full = PCA(n_components=max_possible_components)
                    pca_full.fit(X_class)

                    cumulative_variance_full = np.cumsum(pca_full.explained_variance_ratio_)

                    selected_components = int(
                        np.argmax(cumulative_variance_full >= (simca_variance_threshold / 100)) + 1
                    )

                    loo_errors = []

                else:
                    selected_components = int(min(simca_n_components, max_possible_components))
                    loo_errors = []

                pca_model = PCA(n_components=selected_components)
                scores = pca_model.fit_transform(X_class)

                t2_values, q_values = calculate_simca_diagnostics(X_class, pca_model)

                if simca_validation_limit_strategy == "max":
                    t2_limit = np.max(t2_values)
                    q_limit = np.max(q_values)

                else:
                    t2_limit = np.percentile(t2_values,simca_confidence_level)
                    q_limit = np.percentile(q_values,simca_confidence_level)

                simca_models[cls] = {
                    "pca_model": pca_model,
                    "t2_limit": t2_limit,
                    "q_limit": q_limit,
                }

                class_ids = np.array(id_simca)[class_simca_array == cls]

                for i in range(X_class.shape[0]):
                    accepted = (
                        t2_values[i] <= t2_limit
                        and q_values[i] <= q_limit
                    )

                    simca_diagnostics.append({
                        "sample_id": str(class_ids[i]),
                        "class": str(cls),
                        "hotelling_t2": f"{t2_values[i]:.6f}",
                        "q_residual": f"{q_values[i]:.6f}",
                        "t2_limit": f"{t2_limit:.6f}",
                        "q_limit": f"{q_limit:.6f}",
                        "accepted": "Yes" if accepted else "No",
                    })

                explained_variance = pca_model.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)

                class_ids = np.array(id_simca)[class_simca_array == cls]

                fig_scores = px.scatter(
                    x=scores[:, 0],
                    y=scores[:, 1],
                    hover_name=class_ids,
                    title=f"SIMCA Score Plot - Class {cls}",
                    labels={
                        "x": f"PC1 ({explained_variance[0] * 100:.2f}%)",
                        "y": f"PC2 ({explained_variance[1] * 100:.2f}%)" if len(explained_variance) > 1 else "PC2",
                    }
                )

                fig_scores.update_layout(title_x=0.5)

                explained_variance = pca_model.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)

                fig_t2_q = px.scatter(
                    x=t2_values,
                    y=q_values,
                    hover_name=class_ids,
                    title=f"T² vs Q Residual - Class {cls}",
                    labels={
                        "x": "Hotelling T²",
                        "y": "Q Residual (DModX)"
                    }
                )

                fig_t2_q.add_vline(
                    x=t2_limit,
                    line_dash="dash",
                    line_color="red"
                )

                fig_t2_q.add_hline(
                    y=q_limit,
                    line_dash="dash",
                    line_color="red"
                )

                fig_t2_q.update_layout(
                    title_x=0.5,
                    height=650
                )

                simca_models_summary.append({
                    "class": cls,
                    "components": int(selected_components),
                    "samples": int(X_class.shape[0]),
                    "variables": int(X_class.shape[1]),
                    "pc_selection_method": (
                        "Leave One Out" if simca_pc_selection_method == "loo"
                        else f"Explained variance ≥ {simca_variance_threshold:.1f}%"
                        if simca_pc_selection_method == "variance"
                        else "Manual"
                    ),
                    "loo_errors": [
                        {
                            "components": row["components"],
                            "mean_reconstruction_error": f"{row['mean_reconstruction_error']:.6f}",
                        }
                        for row in loo_errors
                    ],
                    "pc1_variance": f"{explained_variance[0] * 100:.2f}",
                    "pc2_variance": f"{explained_variance[1] * 100:.2f}" if len(explained_variance) > 1 else "",
                    "cumulative_variance": f"{cumulative_variance[-1] * 100:.2f}",
                    "score_plot": fig_scores.to_html(full_html=False),
                    "t2_limit": f"{t2_limit:.6f}",
                    "q_limit": f"{q_limit:.6f}",
                    "confidence_level":simca_confidence_level,
                    "t2_q_plot": fig_t2_q.to_html(full_html=False),
                })

            if data_simca_external.strip():
                external_lines = [
                    line.strip()
                    for line in data_simca_external.splitlines()
                    if line.strip()
                ]

                external_total = 0
                external_correct = 0
                external_rejected = 0
                external_multiple = 0
                y_true_external = []
                y_pred_external = []

                for line in external_lines:

                    values = re.split(r'[\t,;]+', line)

                    if use_first_column_as_id_simca:
                        sample_id = str(values[0])
                        values = values[1:]
                    else:
                        sample_id = str(external_total + 1)

                    if external_has_class_simca:
                        true_class = str(values[-1])
                        values = values[:-1]
                    else:
                        true_class = None

                    try:

                        sample_values = np.array(
                            [float(v) for v in values],
                            dtype=float
                        ).reshape(1, -1)

                    except ValueError:
                        continue

                    if sample_values.shape[1] != data_columns.shape[1]:
                        continue

                    if simca_scaler is not None:
                        sample_values = simca_scaler.transform(
                            sample_values
                        )

                    accepted_models, predicted_class = evaluate_simca_sample(
                        sample_values,
                        simca_models
                    )

                    if true_class is not None:
                        y_true_external.append(str(true_class))
                        y_pred_external.append(str(predicted_class))

                    external_total += 1

                    if predicted_class == "Rejected":
                        external_rejected += 1

                    if predicted_class == "Multiple":
                        external_multiple += 1

                    correct = None

                    if true_class is not None:

                        correct = (
                            predicted_class == true_class
                        )

                        if correct:
                            external_correct += 1

                    simca_external_classification.append({
                        "sample_id": sample_id,
                        "true_class": true_class,
                        "accepted_models": ", ".join(accepted_models) if accepted_models else "None",
                        "predicted_class": predicted_class,
                        "correct": "Yes" if correct else "No" if correct is not None else "",
                    })
                
                if external_has_class_simca and len(y_true_external) > 0:
                    classes_cm = sorted(
                        list(
                            set(y_true_external)
                            | set(y_pred_external)
                        )
                    )

                    if "Rejected" not in classes_cm:
                        classes_cm.append("Rejected")

                    if "Multiple" not in classes_cm:
                        classes_cm.append("Multiple")

                    cm_rows = []

                    for true_cls in classes_cm:

                        row_counts = []
                        for pred_cls in classes_cm:
                            count = sum(
                                1
                                for yt, yp in zip(y_true_external, y_pred_external)
                                if yt == true_cls and yp == pred_cls
                            )

                            row_counts.append(count)

                        cm_rows.append({
                            "true_class": true_cls,
                            "counts": row_counts,
                        })

                    simca_external_confusion_matrix = {
                        "classes": classes_cm,
                        "rows": cm_rows,
                    }

                    sensitivity_specificity_rows = []
                    total_samples_cm = len(y_true_external)
                    for cls in classes_cm:

                        if cls in ["Rejected", "Multiple"]:
                            continue

                        tp = sum(
                            1
                            for yt, yp in zip(y_true_external, y_pred_external)
                            if yt == cls and yp == cls
                        )

                        fn = sum(
                            1
                            for yt, yp in zip(y_true_external, y_pred_external)
                            if yt == cls and yp != cls
                        )

                        fp = sum(
                            1
                            for yt, yp in zip(y_true_external, y_pred_external)
                            if yt != cls and yp == cls
                        )

                        tn = total_samples_cm - tp - fn - fp

                        sensitivity = (
                            tp / (tp + fn) * 100
                            if (tp + fn) > 0
                            else None
                        )

                        specificity = (
                            tn / (tn + fp) * 100
                            if (tn + fp) > 0
                            else None
                        )

                        sensitivity_specificity_rows.append({
                            "class": cls,
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "tn": tn,
                            "sensitivity":
                                f"{sensitivity:.2f}"
                                if sensitivity is not None
                                else "N/A",
                            "specificity":
                                f"{specificity:.2f}"
                                if specificity is not None
                                else "N/A",
                        })
                    
                    simca_external_sensitivity_specificity = (sensitivity_specificity_rows)

                simca_external_summary = {
                    "total_samples": external_total,
                    "correct_classifications": external_correct,
                    "rejected_samples": external_rejected,
                    "multiple_assignments": external_multiple,
                    "overall_accuracy":
                        f"{(external_correct / external_total) * 100:.2f}"
                        if external_has_class_simca and external_total > 0
                        else None,
                }

            for i in range(data_columns.shape[0]):

                sample = data_columns[i:i + 1]
                sample_id = str(id_simca[i])
                true_class = str(class_simca[i])

                accepted_models = []

                for model_class, model_info in simca_models.items():

                    pca_model = model_info["pca_model"]
                    t2_limit = model_info["t2_limit"]
                    q_limit = model_info["q_limit"]

                    t2_sample, q_sample = calculate_simca_diagnostics(sample, pca_model)

                    accepted = (
                        t2_sample[0] <= t2_limit
                        and q_sample[0] <= q_limit
                    )

                    if accepted:
                        accepted_models.append(str(model_class))

                if len(accepted_models) == 1:
                    predicted_class = accepted_models[0]
                elif len(accepted_models) == 0:
                    predicted_class = "Rejected"
                else:
                    predicted_class = "Multiple"

                correct = predicted_class == true_class

                simca_classification.append({
                    "sample_id": sample_id,
                    "true_class": true_class,
                    "accepted_models": ", ".join(accepted_models) if accepted_models else "None",
                    "predicted_class": predicted_class,
                    "correct": "Yes" if correct else "No",
                })

            training_summary = []

            total_samples = len(simca_classification)
            correct_classifications = 0
            rejected_samples = 0
            multiple_assignments = 0

            for cls in sorted(class_counts.keys()):
                class_rows = [
                    row for row in simca_classification
                    if row["true_class"] == cls
                ]

                accepted_count = sum(
                    1 for row in class_rows
                    if row["predicted_class"] == cls
                )

                rejected_count = sum(
                    1 for row in class_rows
                    if row["predicted_class"] == "Rejected"
                )

                multiple_count = sum(
                    1 for row in class_rows
                    if row["predicted_class"] == "Multiple"
                )

                total_class = len(class_rows)

                training_summary.append({
                    "class": cls,
                    "samples": total_class,
                    "accepted": accepted_count,
                    "rejected": rejected_count,
                    "multiple": multiple_count,
                    "acceptance_rate": f"{(accepted_count / total_class) * 100:.2f}" if total_class > 0 else "0.00",
                })

            correct_classifications = sum(
                1 for row in simca_classification
                if row["correct"] == "Yes"
            )

            rejected_samples = sum(
                1 for row in simca_classification
                if row["predicted_class"] == "Rejected"
            )

            multiple_assignments = sum(
                1 for row in simca_classification
                if row["predicted_class"] == "Multiple"
            )

            overall_summary = {
                "total_samples": total_samples,
                "correct_classifications": correct_classifications,
                "rejected_samples": rejected_samples,
                "multiple_assignments": multiple_assignments,
                "overall_accuracy": f"{(correct_classifications / total_samples) * 100:.2f}" if total_samples > 0 else "0.00",
            }

            simca_coomans_class_x = request.POST.get('simca_coomans_class_x', '')
            simca_coomans_class_y = request.POST.get('simca_coomans_class_y', '')

            available_classes = sorted(list(class_counts.keys()))

            if not simca_coomans_class_x and len(available_classes) >= 2:
                simca_coomans_class_x = available_classes[0]

            if not simca_coomans_class_y and len(available_classes) >= 2:
                simca_coomans_class_y = available_classes[1]

            context['simca_coomans_class_x'] = simca_coomans_class_x
            request.session['simca_coomans_class_x'] = simca_coomans_class_x

            context['simca_coomans_class_y'] = simca_coomans_class_y
            request.session['simca_coomans_class_y'] = simca_coomans_class_y

            simca_coomans_plot = None
            if (
                simca_coomans_class_x in simca_models
                and simca_coomans_class_y in simca_models
                and simca_coomans_class_x != simca_coomans_class_y
            ):
                model_x = simca_models[simca_coomans_class_x]["pca_model"]
                model_y = simca_models[simca_coomans_class_y]["pca_model"]

                q_limit_x = simca_models[simca_coomans_class_x]["q_limit"]
                q_limit_y = simca_models[simca_coomans_class_y]["q_limit"]

                _, q_x = calculate_simca_diagnostics(data_columns, model_x)
                _, q_y = calculate_simca_diagnostics(data_columns, model_y)

                fig_coomans = px.scatter(
                    x=q_x,
                    y=q_y,
                    color=class_simca,
                    hover_name=id_simca,
                    title=f"Cooman's Plot - {simca_coomans_class_x} vs {simca_coomans_class_y}",
                    labels={
                        "x": f"DModX / Q residual to {simca_coomans_class_x}",
                        "y": f"DModX / Q residual to {simca_coomans_class_y}",
                        "color": "Class",
                    }
                )

                fig_coomans.add_vline(x=q_limit_x, line_dash="dash", line_color="red")
                fig_coomans.add_hline(y=q_limit_y, line_dash="dash", line_color="red")
                fig_coomans.update_layout(title_x=0.5, height=650)

                simca_coomans_plot = fig_coomans.to_html(full_html=False)

            simca_cv_classification = []
            simca_cv_summary = None
            simca_split_classification = []
            simca_split_summary = None
            if simca_validation_mode == "loo":
                for left_out_idx in range(data_columns.shape[0]):

                    X_train_cv = np.delete(data_columns, left_out_idx, axis=0)
                    y_train_cv = np.delete(np.array(class_simca), left_out_idx)

                    sample = data_columns[left_out_idx:left_out_idx + 1]
                    sample_id = str(id_simca[left_out_idx])
                    true_class = str(class_simca[left_out_idx])

                    cv_class_counts = Counter(y_train_cv)
                    cv_class_array = np.array(y_train_cv)

                    cv_models = {}

                    for cls_cv in sorted(cv_class_counts.keys()):
                        X_class_cv = X_train_cv[cv_class_array == cls_cv]

                        max_possible_components_cv = min(
                            X_class_cv.shape[0],
                            X_class_cv.shape[1]
                        )

                        if max_possible_components_cv < 1:
                            continue

                        if simca_pc_selection_method == "loo":
                            selected_components_cv, _ = select_simca_components_loo(X_class_cv)
                            selected_components_cv = int(selected_components_cv)

                        elif simca_pc_selection_method == "variance":
                            pca_full_cv = PCA(n_components=max_possible_components_cv)
                            pca_full_cv.fit(X_class_cv)

                            cumulative_variance_cv = np.cumsum(
                                pca_full_cv.explained_variance_ratio_
                            )

                            selected_components_cv = int(
                                np.argmax(
                                    cumulative_variance_cv >= (simca_variance_threshold / 100)
                                ) + 1
                            )

                        else:
                            selected_components_cv = int(
                                min(simca_n_components, max_possible_components_cv)
                            )

                        pca_cv = PCA(n_components=selected_components_cv)
                        pca_cv.fit(X_class_cv)

                        t2_cv, q_cv = calculate_simca_diagnostics(X_class_cv, pca_cv)

                        if simca_validation_limit_strategy == "max":
                            t2_limit_cv = np.max(t2_cv)
                            q_limit_cv = np.max(q_cv)
                        else:
                            t2_limit_cv = np.percentile(t2_cv, simca_confidence_level)
                            q_limit_cv = np.percentile(q_cv, simca_confidence_level)

                        cv_models[str(cls_cv)] = {
                            "pca_model": pca_cv,
                            "t2_limit": t2_limit_cv,
                            "q_limit": q_limit_cv,
                        }

                    accepted_models, predicted_class = evaluate_simca_sample(
                        sample,
                        cv_models
                    )

                    simca_cv_classification.append({
                        "sample_id": sample_id,
                        "true_class": true_class,
                        "accepted_models": ", ".join(accepted_models) if accepted_models else "None",
                        "predicted_class": predicted_class,
                        "correct": "Yes" if predicted_class == true_class else "No",
                    })

                cv_total = len(simca_cv_classification)
                cv_correct = sum(
                    1 for row in simca_cv_classification
                    if row["correct"] == "Yes"
                )
                cv_rejected = sum(
                    1 for row in simca_cv_classification
                    if row["predicted_class"] == "Rejected"
                )
                cv_multiple = sum(
                    1 for row in simca_cv_classification
                    if row["predicted_class"] == "Multiple"
                )

                simca_cv_summary = {
                    "method": "Leave One Out",
                    "total_samples": cv_total,
                    "correct_classifications": cv_correct,
                    "rejected_samples": cv_rejected,
                    "multiple_assignments": cv_multiple,
                    "overall_accuracy": f"{(cv_correct / cv_total) * 100:.2f}" if cv_total > 0 else "0.00",
                }

            if simca_validation_mode == "split":

                indices = np.arange(data_columns.shape[0])

                train_idx, test_idx = train_test_split(
                    indices,
                    test_size=0.20,
                    random_state=42,
                    stratify=class_simca
                )

                X_train_split = data_columns[train_idx]
                y_train_split = np.array(class_simca)[train_idx]

                X_test_split = data_columns[test_idx]
                y_test_split = np.array(class_simca)[test_idx]
                id_test_split = np.array(id_simca)[test_idx]

                split_class_counts = Counter(y_train_split)
                split_class_array = np.array(y_train_split)

                split_models = {}

                for cls_split in sorted(split_class_counts.keys()):

                    X_class_split = X_train_split[
                        split_class_array == cls_split
                    ]

                    max_possible_components_split = min(
                        X_class_split.shape[0],
                        X_class_split.shape[1]
                    )

                    if max_possible_components_split < 1:
                        continue

                    if simca_pc_selection_method == "loo":
                        selected_components_split, _ = select_simca_components_loo(
                            X_class_split
                        )
                        selected_components_split = int(selected_components_split)

                    elif simca_pc_selection_method == "variance":
                        pca_full_split = PCA(
                            n_components=max_possible_components_split
                        )
                        pca_full_split.fit(X_class_split)

                        cumulative_variance_split = np.cumsum(
                            pca_full_split.explained_variance_ratio_
                        )

                        selected_components_split = int(
                            np.argmax(
                                cumulative_variance_split >= (
                                    simca_variance_threshold / 100
                                )
                            ) + 1
                        )

                    else:
                        selected_components_split = int(
                            min(
                                simca_n_components,
                                max_possible_components_split
                            )
                        )

                    pca_split = PCA(
                        n_components=selected_components_split
                    )
                    pca_split.fit(X_class_split)

                    t2_split, q_split = calculate_simca_diagnostics(
                        X_class_split,
                        pca_split
                    )

                    if simca_validation_limit_strategy == "max":
                        t2_limit_split = np.max(t2_split)
                        q_limit_split = np.max(q_split)
                    else:
                        t2_limit_split = np.percentile(t2_split, simca_confidence_level)
                        q_limit_split = np.percentile(q_split, simca_confidence_level)

                    split_models[str(cls_split)] = {
                        "pca_model": pca_split,
                        "t2_limit": t2_limit_split,
                        "q_limit": q_limit_split,
                    }

                for i in range(X_test_split.shape[0]):

                    sample = X_test_split[i:i + 1]
                    sample_id = str(id_test_split[i])
                    true_class = str(y_test_split[i])

                    accepted_models, predicted_class = evaluate_simca_sample(
                        sample,
                        split_models
                    )

                    simca_split_classification.append({
                        "sample_id": sample_id,
                        "true_class": true_class,
                        "accepted_models": ", ".join(accepted_models)
                            if accepted_models else "None",
                        "predicted_class": predicted_class,
                        "correct": "Yes"
                            if predicted_class == true_class else "No",
                    })

                split_total = len(simca_split_classification)

                split_correct = sum(
                    1 for row in simca_split_classification
                    if row["correct"] == "Yes"
                )

                split_rejected = sum(
                    1 for row in simca_split_classification
                    if row["predicted_class"] == "Rejected"
                )

                split_multiple = sum(
                    1 for row in simca_split_classification
                    if row["predicted_class"] == "Multiple"
                )

                simca_split_summary = {
                    "method": "Random split 80/20",
                    "training_samples": len(train_idx),
                    "validation_samples": len(test_idx),
                    "correct_classifications": split_correct,
                    "rejected_samples": split_rejected,
                    "multiple_assignments": split_multiple,
                    "overall_accuracy": f"{(split_correct / split_total) * 100:.2f}"
                        if split_total > 0 else "0.00",
                }

            results_simca = {
                "num_samples": len(id_simca),
                "num_variables": data_columns.shape[1],
                "num_classes": len(class_counts),
                "classes": sorted(class_counts.keys()),
                "class_counts": dict(class_counts),
                "models_summary": simca_models_summary,
                "diagnostics": simca_diagnostics,
                "classification": simca_classification,
                "training_summary": training_summary,
                "overall_summary": overall_summary,
                "coomans_plot": simca_coomans_plot,
                "coomans_class_x": simca_coomans_class_x,
                "coomans_class_y": simca_coomans_class_y,
                "cv_classification": simca_cv_classification,
                "cv_summary": simca_cv_summary,
                "validation_mode": simca_validation_mode,
                "split_classification": simca_split_classification,
                "split_summary": simca_split_summary,
                "external_classification": simca_external_classification,
                "external_summary": simca_external_summary,
                "external_confusion_matrix":simca_external_confusion_matrix,
                "external_sensitivity_specificity":simca_external_sensitivity_specificity,
            }

            context['results_simca'] = results_simca
            request.session['results_simca'] = results_simca

            context['headers_simca'] = headers_simca
            request.session['headers_simca'] = headers_simca

            context['id_simca'] = id_simca
            request.session['id_simca'] = id_simca

            context['class_simca'] = class_simca
            request.session['class_simca'] = class_simca

        except ValueError as e:
            context['error_simca'] = str(e)
            context['results_simca'] = None

###########################################################################################

    return render(request, "multivariate/multivariate.html", context)