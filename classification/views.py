from django.shortcuts import render
import numpy as np
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

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

def parse_classification_data(raw_data, use_first_row_as_header=False, use_first_column_as_id=False, require_class=True):
    raw_data = raw_data.replace("\r", "").strip()
    rows = [re.split(r"[\t,;]+", row.strip()) for row in raw_data.split("\n") if row.strip()]

    if not rows:
        raise ValueError("Please enter data before calculating.")

    headers = []
    sample_ids = []
    class_labels = []

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

    if require_class:
        class_labels = [str(row[-1]) for row in rows]
        rows = [row[:-1] for row in rows]

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
                    raise ValueError("The first row seems to contain non-numeric values, but 'Use first row as variable names' is not checked.")
                raise ValueError("Non-numeric value found. Please make sure all data entries are valid numbers.")

        numeric_rows.append(numeric_row)
        row_lengths.append(len(numeric_row))

    if len(set(row_lengths)) > 1:
        raise ValueError("Missing value/s detected. Please ensure all rows have the same number of variables.")

    X = np.array(numeric_rows, dtype=float)

    if X.shape[1] < 1:
        raise ValueError("At least one numeric variable is required.")

    if headers:
        if use_first_column_as_id:
            headers = headers[1:]
        if require_class and len(headers) > X.shape[1]:
            headers = headers[:-1]

    if not headers:
        headers = [f"var {i + 1}" for i in range(X.shape[1])]

    if require_class and len(set(class_labels)) < 2:
        raise ValueError("At least two classes/groups are required for classification.")

    return X, np.array(class_labels), sample_ids, headers

class PLSDAClassifier:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = None
        self.encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, X, y):
        y_encoded = self.encoder.fit_transform(y)
        self.classes_ = self.encoder.classes_
        Y = np.zeros((len(y_encoded), len(self.classes_)))
        Y[np.arange(len(y_encoded)), y_encoded] = 1
        max_components = min(self.n_components, X.shape[0] - 1, X.shape[1])
        max_components = max(1, max_components)
        self.model = PLSRegression(n_components=max_components)
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        scores = self.model.predict(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        scores = self.model.predict(X)
        scores = np.maximum(scores, 0)
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return scores / row_sums

    def transform(self, X):
        return self.model.transform(X)

def build_model(method,n_components=2,knn_n_neighbors=5):

    if method == "lda":
        return LinearDiscriminantAnalysis()

    if method == "qda":
        return QuadraticDiscriminantAnalysis()

    if method == "plsda":
        return PLSDAClassifier(
            n_components=n_components
        )

    if method == "svm":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        )

    if method == "logreg":
        return LogisticRegression(
            max_iter=5000,
            random_state=42
        )

    if method == "naivebayes":
        return GaussianNB()

    if method == "knn":
        return KNeighborsClassifier(
            n_neighbors=knn_n_neighbors
        )

    raise ValueError("Invalid classification method selected.")

def safe_predict_proba(model, X, classes):
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            model_classes = list(model.classes_)
            full = np.zeros((X.shape[0], len(classes)))
            for j, cls in enumerate(classes):
                if cls in model_classes:
                    full[:, j] = proba[:, model_classes.index(cls)]
            return full
        except Exception:
            return None
    return None

def classification_rows(model, X, y_true, sample_ids, classes):
    y_pred = model.predict(X)
    proba = safe_predict_proba(model, X, classes)
    rows = []
    for i, sample_id in enumerate(sample_ids):
        prob_values = []
        if proba is not None:
            prob_values = [f"{100 * proba[i, j]:.2f}" for j in range(len(classes))]
        rows.append({
            "sample_id": str(sample_id),
            "true_class": str(y_true[i]) if y_true is not None else "",
            "predicted_class": str(y_pred[i]),
            "correct": "Yes" if y_true is not None and str(y_pred[i]) == str(y_true[i]) else "No" if y_true is not None else "",
            "probabilities": prob_values,
        })
    return rows

def summarize_performance(rows, classes):
    if not rows or not rows[0].get("true_class"):
        return None, None, None

    y_true = [row["true_class"] for row in rows]
    y_pred = [row["predicted_class"] for row in rows]
    labels = list(classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_rows = []
    for i, cls in enumerate(labels):
        cm_rows.append({"true_class": cls, "counts": [int(v) for v in cm[i]]})

    total = len(rows)
    correct = sum(1 for row in rows if row["correct"] == "Yes")

    by_class = []
    for cls in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        tn = total - tp - fn - fp
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) else None
        specificity = tn / (tn + fp) * 100 if (tn + fp) else None
        precision = tp / (tp + fp) * 100 if (tp + fp) else None
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision is not None and sensitivity is not None and (precision + sensitivity) else None
        by_class.append({
            "class": cls,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "sensitivity": f"{sensitivity:.2f}" if sensitivity is not None else "N/A",
            "specificity": f"{specificity:.2f}" if specificity is not None else "N/A",
            "precision": f"{precision:.2f}" if precision is not None else "N/A",
            "f1": f"{f1:.2f}" if f1 is not None else "N/A",
        })

    summary = {
        "total_samples": total,
        "correct_classifications": correct,
        "overall_accuracy": f"{(correct / total) * 100:.2f}" if total else "0.00",
    }

    confusion = {"classes": labels, "rows": cm_rows}
    return summary, confusion, by_class

def make_scores_plot(method, model, X, y, sample_ids):
    try:
        if method == "lda":
            scores = model.transform(X)
            x = scores[:, 0]
            y_axis = scores[:, 1] if scores.shape[1] > 1 else np.zeros(scores.shape[0])
            x_label = "LD1"
            y_label = "LD2" if scores.shape[1] > 1 else "LD2"
            title = "LDA Scores Plot"
        elif method == "plsda":
            scores = model.transform(X)
            x = scores[:, 0]
            y_axis = scores[:, 1] if scores.shape[1] > 1 else np.zeros(scores.shape[0])
            x_label = "LV1"
            y_label = "LV2" if scores.shape[1] > 1 else "LV2"
            title = "PLS-DA Scores Plot"
        else:
            return None

        fig = px.scatter(x=x, y=y_axis, color=y, hover_name=sample_ids, title=title, labels={"x": x_label, "y": y_label, "color": "Class"})
        fig.update_layout(title_x=0.5, height=650)
        return fig.to_html(full_html=False)
    except Exception:
        return None

def make_confusion_heatmap(confusion, title):
    if not confusion:
        return None
    z = [row["counts"] for row in confusion["rows"]]
    labels = confusion["classes"]
    fig = go.Figure(data=go.Heatmap(z=z, x=labels, y=labels, text=z, texttemplate="%{text}"))
    fig.update_layout(title=title, title_x=0.5, xaxis_title="Predicted Class", yaxis_title="Actual Class", height=550)
    return fig.to_html(full_html=False)


def calculate_plsda_explained_variance(pls_model, X):
    """
    Returns per-component and cumulative explained X variance for a fitted PLSRegression model.
    The per-component value is calculated from the incremental X reconstruction sum of squares.
    """
    T = pls_model.x_scores_
    P = pls_model.x_loadings_

    total_ss = np.sum(X ** 2)
    if total_ss == 0:
        return []

    previous_ss = 0.0
    rows = []

    for i in range(T.shape[1]):
        T_i = T[:, : i + 1]
        P_i = P[:, : i + 1]
        X_hat = T_i @ P_i.T

        cumulative_ss = np.sum(X_hat ** 2)
        explained = max(0.0, cumulative_ss - previous_ss)
        previous_ss = cumulative_ss

        rows.append({
            "lv": i + 1,
            "explained": f"{(explained / total_ss) * 100:.2f}",
            "cumulative": f"{(cumulative_ss / total_ss) * 100:.2f}",
        })

    return rows


def calculate_vip(pls_model):
    """
    Variable Importance in Projection (VIP) scores for a fitted PLSRegression model.
    """
    T = pls_model.x_scores_
    W = pls_model.x_weights_
    Q = pls_model.y_loadings_

    p, h = W.shape

    s = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    total_s = np.sum(s)

    if total_s == 0:
        return np.zeros(p)

    vip = np.zeros(p)

    for i in range(p):
        weight = np.array([
            (W[i, j] / np.linalg.norm(W[:, j])) ** 2 * s[j]
            for j in range(h)
        ])
        vip[i] = np.sqrt(p * np.sum(weight) / total_s)

    return vip


def make_vip_plot(vip_table):
    if not vip_table:
        return None

    fig = px.bar(
        x=[row["variable"] for row in vip_table],
        y=[row["vip"] for row in vip_table],
        title="VIP Scores",
        labels={"x": "Variable", "y": "VIP Score"},
    )
    fig.add_hline(y=1, line_dash="dash")
    fig.update_layout(title_x=0.5, height=550)

    return fig.to_html(full_html=False)

def classification(request):
    tab = request.GET.get("tab", "lda")
    method = tab

    data_key = f"data_classification_{method}"
    external_key = f"data_classification_external_{method}"
    results_key = f"results_classification_{method}"
    header_key = f"use_first_row_as_header_classification_{method}"
    id_key = f"use_first_column_as_id_classification_{method}"
    external_has_class_key = f"external_has_class_classification_{method}"
    normalization_key = f"normalization_classification_{method}"
    validation_key = f"classification_validation_mode_{method}"
    plsda_components_key = f"plsda_n_components_{method}"
    knn_key = f"knn_n_neighbors_{method}"

    context = {
        "segment": "classification",
        "active_tab": tab,
        "data_classification": request.session.get(data_key, ""),
        "data_classification_external": request.session.get(external_key, ""),
        "use_first_row_as_header_classification": "checked" if request.session.get(header_key, False) else "",
        "use_first_column_as_id_classification": "checked" if request.session.get(id_key, False) else "",
        "external_has_class_classification": "checked" if request.session.get(external_has_class_key, False) else "",
        "normalization_classification": request.session.get(normalization_key, "zscore"),
        "classification_validation_mode": request.session.get(validation_key, "training"),
        "plsda_n_components": request.session.get(plsda_components_key, 2),
        "results_classification": request.session.get(results_key, None),
        "knn_n_neighbors": request.session.get(knn_key,5),
    }

    if request.method == "POST" and request.POST.get("clear_classification") == "true":
        for key in [
            data_key,
            external_key,
            header_key,
            id_key,
            external_has_class_key,
            normalization_key,
            validation_key,
            plsda_components_key,
            knn_key,
            results_key,
        ]:
            request.session.pop(key, None)
        context.update({
            "data_classification": "",
            "data_classification_external": "",
            "use_first_row_as_header_classification": "",
            "use_first_column_as_id_classification": "",
            "external_has_class_classification": "",
            "normalization_classification": "zscore",
            "classification_validation_mode": "training",
            "plsda_n_components": 2,
            "results_classification": None,
        })
        return render(request, "classification/classification.html", context)

    if request.method == "POST" and tab in ["lda", "qda", "plsda", "svm", "logreg", "naivebayes", "knn"]:
        method = tab
        data = request.POST.get("data_classification", "")
        data_external = request.POST.get("data_classification_external", "")
        use_header = request.POST.get("use_first_row_as_header_classification") == "on"
        use_id = request.POST.get("use_first_column_as_id_classification") == "on"
        external_has_class = request.POST.get("external_has_class_classification") == "on"
        normalization = request.POST.get("normalization_classification", "zscore")
        validation_mode = request.POST.get("classification_validation_mode", "training")

        try:
            plsda_n_components = int(request.POST.get("plsda_n_components", 2))
        except ValueError:
            plsda_n_components = 2
        plsda_n_components = max(1, plsda_n_components)

        try:
            knn_n_neighbors = int(
                request.POST.get(
                    "knn_n_neighbors",
                    5
                )
            )
        except ValueError:
            knn_n_neighbors = 5

        knn_n_neighbors = max(1, min(50, knn_n_neighbors))

        for key, value in {
            data_key: data,
            external_key: data_external,
            header_key: use_header,
            id_key: use_id,
            external_has_class_key: external_has_class,
            normalization_key: normalization,
            validation_key: validation_mode,
            plsda_components_key: plsda_n_components,
            knn_key: knn_n_neighbors,
        }.items():
            request.session[key] = value

        context.update({
            "data_classification": data,
            "data_classification_external": data_external,
            "use_first_row_as_header_classification": "checked" if use_header else "",
            "use_first_column_as_id_classification": "checked" if use_id else "",
            "external_has_class_classification": "checked" if external_has_class else "",
            "normalization_classification": normalization,
            "classification_validation_mode": validation_mode,
            "plsda_n_components": plsda_n_components,
        })

        try:
            X, y, sample_ids, headers = parse_classification_data(data, use_header, use_id, require_class=True)
            X_scaled, scaler = normalize_data(X, normalization)
            classes = sorted(list(set(y)))
            class_counts = Counter(y)

            model = build_model(
                method,
                plsda_n_components,
                knn_n_neighbors
            )
            model.fit(X_scaled, y)

            plsda_variance = []
            vip_table = []
            vip_plot = None

            if method == "plsda":
                scores = model.model.x_scores_
                explained_variance = []

                total_ss = np.sum(X_scaled ** 2)

                for i in range(scores.shape[1]):
                    T = model.model.x_scores_[:, : i + 1]
                    P = model.model.x_loadings_[:, : i + 1]

                    X_hat = T @ P.T
                    ss = np.sum(X_hat ** 2)

                    explained_variance.append(100 * ss / total_ss)

                individual_variance = []
                cumulative_variance = []

                previous = 0

                for v in explained_variance:

                    individual_variance.append(v - previous)
                    cumulative_variance.append(v)

                    previous = v

                plsda_variance = []

                for i in range(len(explained_variance)):

                    plsda_variance.append({
                        "lv": i + 1,
                        "explained": f"{individual_variance[i]:.2f}",
                        "cumulative": f"{cumulative_variance[i]:.2f}",
                    })

                # VIP scores
                t = model.model.x_scores_
                w = model.model.x_weights_
                q = model.model.y_loadings_

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
                        "variable": headers[i],
                        "vip": f"{vip_scores[i]:.3f}",
                    }
                    for i in range(len(headers))
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

            training_rows = classification_rows(model, X_scaled, y, sample_ids, classes)
            training_summary, training_confusion, training_by_class = summarize_performance(training_rows, classes)

            score_plot = make_scores_plot(method, model, X_scaled, y, sample_ids)
            confusion_plot = make_confusion_heatmap(training_confusion, "Training Confusion Matrix")

            cv_rows = []
            cv_summary = None
            cv_confusion = None
            cv_by_class = None
            cv_confusion_plot = None

            split_rows = []
            split_summary = None
            split_confusion = None
            split_by_class = None
            split_confusion_plot = None

            if validation_mode == "loo":
                loo = LeaveOneOut()
                for train_index, test_index in loo.split(X_scaled):
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    if len(set(y_train)) < 2:
                        continue
                    cv_model = build_model(method, plsda_n_components)
                    cv_model.fit(X_train, y_train)
                    cv_rows.extend(classification_rows(cv_model, X_test, y_test, [sample_ids[test_index[0]]], classes))
                cv_summary, cv_confusion, cv_by_class = summarize_performance(cv_rows, classes)
                cv_confusion_plot = make_confusion_heatmap(cv_confusion, "LOO Cross-validation Confusion Matrix")

            if validation_mode == "split":
                indices = np.arange(X_scaled.shape[0])
                train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y)
                split_model = build_model(method, plsda_n_components)
                split_model.fit(X_scaled[train_idx], y[train_idx])
                split_ids = [sample_ids[i] for i in test_idx]
                split_rows = classification_rows(split_model, X_scaled[test_idx], y[test_idx], split_ids, classes)
                split_summary, split_confusion, split_by_class = summarize_performance(split_rows, classes)
                if split_summary:
                    split_summary["training_samples"] = len(train_idx)
                    split_summary["validation_samples"] = len(test_idx)
                split_confusion_plot = make_confusion_heatmap(split_confusion, "Random Split Validation Confusion Matrix")

            external_rows = []
            external_summary = None
            external_confusion = None
            external_by_class = None
            external_confusion_plot = None

            if data_external.strip():
                X_ext, y_ext, ext_ids, _ = parse_classification_data(data_external,use_header,use_id,require_class=external_has_class)
                if X_ext.shape[1] != X.shape[1]:
                    raise ValueError("External validation data must have the same number of numeric variables as the training data.")
                X_ext_scaled = scaler.transform(X_ext) if scaler is not None else X_ext
                external_rows = classification_rows(model, X_ext_scaled, y_ext if external_has_class else None, ext_ids, classes)
                external_summary, external_confusion, external_by_class = summarize_performance(external_rows, classes)
                external_confusion_plot = make_confusion_heatmap(external_confusion, "External Validation Confusion Matrix")

            method_names = {
                "lda": "LDA",
                "qda": "QDA",
                "plsda": "PLS-DA",
                "svm": "SVM",
                "logreg": "Logistic Regression",
                "naivebayes": "Naive Bayes",
                "knn": "k-NN",
            }

            results = {
                "method": method_names[method],
                "num_samples": len(sample_ids),
                "num_variables": X.shape[1],
                "num_classes": len(classes),
                "classes": classes,
                "class_counts": dict(class_counts),
                "probability_classes": classes,
                "training_classification": training_rows,
                "training_summary": training_summary,
                "training_confusion_matrix": training_confusion,
                "training_by_class": training_by_class,
                "score_plot": score_plot,
                "confusion_plot": confusion_plot,
                "cv_classification": cv_rows,
                "cv_summary": cv_summary,
                "cv_confusion_matrix": cv_confusion,
                "cv_by_class": cv_by_class,
                "cv_confusion_plot": cv_confusion_plot,
                "split_classification": split_rows,
                "split_summary": split_summary,
                "split_confusion_matrix": split_confusion,
                "split_by_class": split_by_class,
                "split_confusion_plot": split_confusion_plot,
                "external_classification": external_rows,
                "external_summary": external_summary,
                "external_confusion_matrix": external_confusion,
                "external_by_class": external_by_class,
                "external_confusion_plot": external_confusion_plot,
                "validation_mode": validation_mode,
                "normalization": normalization,
                "n_components": plsda_n_components,
                "plsda_variance": plsda_variance,
                "vip_table": vip_table,
                "vip_plot": vip_plot,
                "knn_n_neighbors": knn_n_neighbors,
            }

            context["results_classification"] = results
            request.session[results_key] = results

        except Exception as e:
            context["error_classification"] = str(e)
            context["results_classification"] = None
            request.session[results_key] = None

    return render(request, "classification/classification.html", context)