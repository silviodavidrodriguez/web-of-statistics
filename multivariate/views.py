from django.shortcuts import render
#from django.contrib.auth.decorators import login_required
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

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

#@login_required(login_url="/login/")
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

    return render(request, "multivariate/multivariate.html", context)