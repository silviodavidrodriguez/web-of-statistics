from django.shortcuts import render
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

def normalize_data(data, method):
    if method == "none":
        return data, None

    scalers = {
        "zscore": StandardScaler(),
        "minmax": MinMaxScaler(),
        "maxabs": MaxAbsScaler(),
        "robust": RobustScaler(),
    }

    if method not in scalers:
        raise ValueError("Invalid normalization method selected.")

    scaler = scalers[method]
    return scaler.fit_transform(data), scaler

def parse_multivar_reg_data(
    raw_data,
    use_first_row_as_header=False,
    use_first_column_as_id=False,
    require_y=True
):
    raw_data = raw_data.replace("\r", "").strip()
    rows = [
        re.split(r"[\t,;]+", row.strip())
        for row in raw_data.split("\n")
        if row.strip()
    ]

    if not rows:
        raise ValueError("Please enter data before calculating.")

    headers = []
    sample_ids = []

    if use_first_row_as_header:
        headers = rows[0]
        rows = rows[1:]

    if not rows:
        raise ValueError("No sample rows found after reading the header.")

    if use_first_column_as_id:
        sample_ids = [str(row[0]) for row in rows]
        rows = [row[1:] for row in rows]
    else:
        sample_ids = [str(i + 1) for i in range(len(rows))]

    numeric_rows = []
    row_lengths = []

    for row_idx, row in enumerate(rows):
        numeric_row = []

        for value in row:
            value = value.strip()

            if not value:
                raise ValueError("Missing value/s detected. Please ensure all values are present.")

            try:
                numeric_row.append(float(value))
            except ValueError:
                if row_idx == 0 and not use_first_row_as_header:
                    raise ValueError(
                        "The first row seems to contain non-numeric values, but 'Use first row as variable names' is not checked."
                    )
                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")

        numeric_rows.append(numeric_row)
        row_lengths.append(len(numeric_row))

    if len(set(row_lengths)) > 1:
        raise ValueError("Missing value/s detected. Please ensure all rows have the same number of columns.")

    data = np.array(numeric_rows, dtype=float)

    if require_y:
        if data.shape[1] < 2:
            raise ValueError("At least one X variable and one response variable Y are required.")

        X = data[:, :-1]
        y = data[:, -1]

        if headers:
            if use_first_column_as_id:
                headers = headers[1:]

            x_headers = headers[:-1]
            y_header = headers[-1]
        else:
            x_headers = [f"X{i + 1}" for i in range(X.shape[1])]
            y_header = "Y"

        return X, y, sample_ids, x_headers, y_header

    else:
        X = data

        if headers:
            if use_first_column_as_id:
                headers = headers[1:]
            x_headers = headers
        else:
            x_headers = [f"X{i + 1}" for i in range(X.shape[1])]

        return X, None, sample_ids, x_headers, None

def calculate_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

def regression_summary(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = calculate_bias(y_true, y_pred)

    return {
        "samples": len(y_true),
        "r2": f"{r2:.4f}",
        "rmse": f"{rmse:.4f}",
        "mae": f"{mae:.4f}",
        "bias": f"{bias:.4f}",
    }

def prediction_rows(y_true, y_pred, sample_ids):
    rows = []

    for i, sample_id in enumerate(sample_ids):
        residual = y_true[i] - y_pred[i]

        rows.append({
            "sample_id": str(sample_id),
            "y_true": f"{y_true[i]:.4f}",
            "y_pred": f"{y_pred[i]:.4f}",
            "residual": f"{residual:.4f}",
        })

    return rows

def make_predicted_plot(y_true, y_pred, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        name="Samples"
    ))

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Ideal",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="Reference Y",
        yaxis_title="Predicted Y",
        height=550,
        template="plotly_white"
    )

    return fig.to_html(full_html=False)

def make_scores_plot(scores, sample_ids, title):
    if scores is None or scores.shape[1] < 1:
        return None

    x = scores[:, 0]
    y = scores[:, 1] if scores.shape[1] > 1 else np.zeros(scores.shape[0])

    fig = px.scatter(
        x=x,
        y=y,
        hover_name=sample_ids,
        title=title,
        labels={
            "x": "Component 1",
            "y": "Component 2",
        }
    )

    fig.update_layout(
        title_x=0.5,
        height=650
    )

    return fig.to_html(full_html=False)

def pcr_explained_variance(pca):
    explained = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)

    rows = []

    for i, value in enumerate(explained):
        rows.append({
            "component": f"PC{i + 1}",
            "explained": f"{value:.2f}",
            "cumulative": f"{cumulative[i]:.2f}",
        })

    return rows

def multivariate_regression(request):
    tab = request.GET.get("tab", "pcr")
    method = tab

    data_key = f"data_multivar_reg_{method}"
    external_key = f"data_multivar_reg_external_{method}"
    results_key = f"results_multivar_reg_{method}"
    header_key = f"use_first_row_as_header_multivar_reg_{method}"
    id_key = f"use_first_column_as_id_multivar_reg_{method}"
    external_has_y_key = f"external_has_y_multivar_reg_{method}"
    normalization_key = f"normalization_multivar_reg_{method}"
    validation_key = f"validation_mode_multivar_reg_{method}"
    components_key = f"n_components_multivar_reg_{method}"

    context = {
        "segment": "multivariate_regression",
        "active_tab": tab,
        "data_multivar_reg": request.session.get(data_key, ""),
        "data_multivar_reg_external": request.session.get(external_key, ""),
        "use_first_row_as_header_multivar_reg": "checked" if request.session.get(header_key, False) else "",
        "use_first_column_as_id_multivar_reg": "checked" if request.session.get(id_key, False) else "",
        "external_has_y_multivar_reg": "checked" if request.session.get(external_has_y_key, False) else "",
        "normalization_multivar_reg": request.session.get(normalization_key, "zscore"),
        "validation_mode_multivar_reg": request.session.get(validation_key, "training"),
        "n_components_multivar_reg": request.session.get(components_key, 2),
        "results_multivar_reg": request.session.get(results_key, None),
    }

    if request.method == "POST" and request.POST.get("clear_multivar_reg") == "true":
        for key in [
            data_key,
            external_key,
            results_key,
            header_key,
            id_key,
            external_has_y_key,
            normalization_key,
            validation_key,
            components_key,
        ]:
            request.session.pop(key, None)

        context.update({
            "data_multivar_reg": "",
            "data_multivar_reg_external": "",
            "use_first_row_as_header_multivar_reg": "",
            "use_first_column_as_id_multivar_reg": "",
            "external_has_y_multivar_reg": "",
            "normalization_multivar_reg": "zscore",
            "validation_mode_multivar_reg": "training",
            "n_components_multivar_reg": 2,
            "results_multivar_reg": None,
        })

        return render(
            request,
            "multivariate_regression/multivariate_regression.html",
            context
        )

    if request.method == "POST" and method in ["pcr", "plsr"]:
        data = request.POST.get("data_multivar_reg", "")
        data_external = request.POST.get("data_multivar_reg_external", "")

        use_header = request.POST.get("use_first_row_as_header_multivar_reg") == "on"
        use_id = request.POST.get("use_first_column_as_id_multivar_reg") == "on"
        external_has_y = request.POST.get("external_has_y_multivar_reg") == "on"

        normalization = request.POST.get("normalization_multivar_reg", "zscore")
        validation_mode = request.POST.get("validation_mode_multivar_reg", "training")

        try:
            n_components = int(request.POST.get("n_components_multivar_reg", 2))
        except ValueError:
            n_components = 2

        n_components = max(1, n_components)

        for key, value in {
            data_key: data,
            external_key: data_external,
            header_key: use_header,
            id_key: use_id,
            external_has_y_key: external_has_y,
            normalization_key: normalization,
            validation_key: validation_mode,
            components_key: n_components,
        }.items():
            request.session[key] = value

        context.update({
            "data_multivar_reg": data,
            "data_multivar_reg_external": data_external,
            "use_first_row_as_header_multivar_reg": "checked" if use_header else "",
            "use_first_column_as_id_multivar_reg": "checked" if use_id else "",
            "external_has_y_multivar_reg": "checked" if external_has_y else "",
            "normalization_multivar_reg": normalization,
            "validation_mode_multivar_reg": validation_mode,
            "n_components_multivar_reg": n_components,
        })

        try:
            X, y, sample_ids, x_headers, y_header = parse_multivar_reg_data(
                data,
                use_header,
                use_id,
                require_y=True
            )

            X_scaled, scaler = normalize_data(X, normalization)

            max_components = min(
                n_components,
                X_scaled.shape[0] - 1,
                X_scaled.shape[1]
            )
            max_components = max(1, max_components)

            max_available_components = min(
                X_scaled.shape[0] - 1,
                X_scaled.shape[1]
            )

            coefficients_table = []
            external_predicted_plot = None
            vip_table = []
            vip_plot = None
            if method == "pcr":
                pca = PCA(n_components=max_components)
                scores = pca.fit_transform(X_scaled)

                reg_model = LinearRegression()
                reg_model.fit(scores, y)

                y_pred = reg_model.predict(scores)

                pcr_coef_scaled = np.dot(
                    pca.components_.T,
                    reg_model.coef_
                )

                coefficients_table = [
                    {
                        "variable": x_headers[i],
                        "coefficient": f"{pcr_coef_scaled[i]:.6f}",
                    }
                    for i in range(len(x_headers))
                ]

                coefficients_table.append({
                    "variable": "Intercept",
                    "coefficient": f"{reg_model.intercept_:.6f}",
                })

                score_plot = make_scores_plot(
                    scores,
                    sample_ids,
                    "PCR Scores Plot"
                )

                explained_variance = pcr_explained_variance(pca)

                predicted_plot = make_predicted_plot(
                    y,
                    y_pred,
                    "PCR Predicted vs Reference"
                )

                vip_table = []
                vip_plot = None

            elif method == "plsr":
                pls_model = PLSRegression(
                    n_components=max_components
                )

                pls_model.fit(X_scaled, y)

                y_pred = pls_model.predict(X_scaled).ravel()

                scores = pls_model.x_scores_

                score_plot = make_scores_plot(
                    scores,
                    sample_ids,
                    "PLSR Scores Plot"
                )

                predicted_plot = make_predicted_plot(
                    y,
                    y_pred,
                    "PLSR Predicted vs Reference"
                )

                explained_variance = []

                total_ss = np.sum(X_scaled ** 2)

                cumulative_variance = []

                for i in range(scores.shape[1]):
                    T = pls_model.x_scores_[:, : i + 1]
                    P = pls_model.x_loadings_[:, : i + 1]

                    X_hat = T @ P.T
                    ss = np.sum(X_hat ** 2)

                    cumulative_variance.append(100 * ss / total_ss)

                previous = 0

                for i, value in enumerate(cumulative_variance):
                    explained_variance.append({
                        "component": f"LV{i + 1}",
                        "explained": f"{value - previous:.2f}",
                        "cumulative": f"{value:.2f}",
                    })

                    previous = value

                coefficients_table = [
                    {
                        "variable": x_headers[i],
                        "coefficient": f"{pls_model.coef_.ravel()[i]:.6f}",
                    }
                    for i in range(len(x_headers))
                ]

                coefficients_table.append({
                    "variable": "Intercept",
                    "coefficient": f"{float(pls_model.intercept_.ravel()[0]):.6f}",
                })

                # VIP scores
                t = pls_model.x_scores_
                w = pls_model.x_weights_
                q = pls_model.y_loadings_

                p = w.shape[0]
                h = w.shape[1]

                s = np.diag(t.T @ t @ q.T @ q)
                total_s = np.sum(s)

                vip_scores = np.zeros(p)

                if total_s != 0:
                    for i in range(p):
                        weight = np.array([
                            (w[i, j] ** 2) * s[j]
                            for j in range(h)
                        ])

                        vip_scores[i] = np.sqrt(
                            p * np.sum(weight) / total_s
                        )

                vip_table = [
                    {
                        "variable": x_headers[i],
                        "vip": f"{vip_scores[i]:.3f}",
                    }
                    for i in range(len(x_headers))
                ]

                vip_table = sorted(
                    vip_table,
                    key=lambda row: float(row["vip"]),
                    reverse=True
                )

                fig_vip = px.bar(
                    x=[row["variable"] for row in vip_table],
                    y=[float(row["vip"]) for row in vip_table],
                    title="VIP Scores",
                    labels={
                        "x": "Variable",
                        "y": "VIP Score",
                    }
                )

                fig_vip.add_hline(y=1, line_dash="dash")
                fig_vip.update_layout(title_x=0.5, height=550)

                vip_plot = fig_vip.to_html(full_html=False)

            training_summary = regression_summary(y, y_pred)
            training_predictions = prediction_rows(y, y_pred, sample_ids)

            cv_summary = None
            split_summary = None
            external_summary = None
            external_predictions = None

            if validation_mode == "loo":
                loo = LeaveOneOut()
                cv_y_true = []
                cv_y_pred = []
                cv_ids = []

                for train_index, test_index in loo.split(X_scaled):
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    max_cv_components = min(
                        max_components,
                        X_train.shape[0] - 1,
                        X_train.shape[1]
                    )
                    max_cv_components = max(1, max_cv_components)

                    if method == "pcr":
                        pca_cv = PCA(n_components=max_cv_components)
                        scores_train = pca_cv.fit_transform(X_train)
                        scores_test = pca_cv.transform(X_test)

                        reg_cv = LinearRegression()
                        reg_cv.fit(scores_train, y_train)

                        pred = reg_cv.predict(scores_test)

                    elif method == "plsr":
                        pls_cv = PLSRegression(n_components=max_cv_components)
                        pls_cv.fit(X_train, y_train)

                        pred = pls_cv.predict(X_test).ravel()

                    cv_y_true.extend(y_test.tolist())
                    cv_y_pred.extend(pred.tolist())
                    cv_ids.append(sample_ids[test_index[0]])

                cv_y_true = np.array(cv_y_true)
                cv_y_pred = np.array(cv_y_pred)

                cv_summary = regression_summary(cv_y_true, cv_y_pred)

            if validation_mode == "split":
                indices = np.arange(X_scaled.shape[0])

                train_idx, test_idx = train_test_split(
                    indices,
                    test_size=0.20,
                    random_state=42
                )

                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                max_split_components = min(
                    max_components,
                    X_train.shape[0] - 1,
                    X_train.shape[1]
                )
                max_split_components = max(1, max_split_components)

                if method == "pcr":
                    pca_split = PCA(n_components=max_split_components)
                    scores_train = pca_split.fit_transform(X_train)
                    scores_test = pca_split.transform(X_test)

                    reg_split = LinearRegression()
                    reg_split.fit(scores_train, y_train)

                    y_split_pred = reg_split.predict(scores_test)

                elif method == "plsr":
                    pls_split = PLSRegression(n_components=max_split_components)
                    pls_split.fit(X_train, y_train)

                    y_split_pred = pls_split.predict(X_test).ravel()

                split_summary = regression_summary(y_test, y_split_pred)
                split_summary["training_samples"] = len(train_idx)
                split_summary["validation_samples"] = len(test_idx)

            if data_external.strip():
                X_ext, y_ext, ext_ids, _, _ = parse_multivar_reg_data(
                    data_external,
                    use_header,
                    use_id,
                    require_y=external_has_y
                )

                if X_ext.shape[1] != X.shape[1]:
                    raise ValueError(
                        "External validation data must have the same number of X variables as the training data."
                    )

                X_ext_scaled = scaler.transform(X_ext) if scaler is not None else X_ext

                if method == "pcr":
                    scores_ext = pca.transform(X_ext_scaled)
                    y_ext_pred = reg_model.predict(scores_ext)

                elif method == "plsr":
                    y_ext_pred = pls_model.predict(X_ext_scaled).ravel()

                if external_has_y:
                    external_summary = regression_summary(y_ext, y_ext_pred)
                    external_predictions = prediction_rows(y_ext, y_ext_pred, ext_ids)

                    external_predicted_plot = make_predicted_plot(
                        y_ext,
                        y_ext_pred,
                        "External Validation Predicted vs Reference"
                    )

                else:
                    external_predictions = [
                        {
                            "sample_id": str(ext_ids[i]),
                            "y_true": "",
                            "y_pred": f"{y_ext_pred[i]:.4f}",
                            "residual": "",
                        }
                        for i in range(len(ext_ids))
                    ]

            method_names = {
                "pcr": "PCR",
                "plsr": "PLSR",
            }

            results = {
                "method": method_names[method],
                "validation_mode": validation_mode,
                "normalization": normalization,
                "n_components": max_components,
                "num_samples": len(sample_ids),
                "num_variables": X.shape[1],
                "response_name": y_header,
                "score_plot": score_plot,
                "predicted_plot": predicted_plot,
                "explained_variance": explained_variance,
                "vip_table": vip_table,
                "vip_plot": vip_plot,
                "training_summary": training_summary,
                "training_predictions": training_predictions,
                "cv_summary": cv_summary,
                "split_summary": split_summary,
                "external_summary": external_summary,
                "external_predictions": external_predictions,
                "max_available_components": max_available_components,
                "external_predicted_plot": external_predicted_plot,
                "coefficients_table": coefficients_table,
            }

            context["results_multivar_reg"] = results
            request.session[results_key] = results

        except Exception as e:
            context["error_multivar_reg"] = str(e)
            context["results_multivar_reg"] = None
            request.session[results_key] = None

    return render(
        request,
        "multivariate_regression/multivariate_regression.html",
        context
    )