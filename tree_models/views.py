from django.shortcuts import render
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import plotly.express as px
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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

def parse_tree_classification_data(
    raw_data,
    use_first_row_as_header=False,
    use_first_column_as_id=False,
    require_class=True
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

    if require_class:
        classes = [str(row[-1]) for row in rows]
        rows = [row[:-1] for row in rows]
    else:
        classes = None

    numeric_rows = []
    row_lengths = []

    for row_idx, row in enumerate(rows):
        numeric_row = []

        for value in row:
            value = value.strip()

            if not value:
                raise ValueError(
                    "Missing value/s detected. Please ensure all values are present."
                )

            try:
                numeric_row.append(float(value))
            except ValueError:
                if row_idx == 0 and not use_first_row_as_header:
                    raise ValueError(
                        "The first row seems to contain non-numeric values, but 'Use first row as variable names' is not checked."
                    )

                raise ValueError(
                    "Non-numeric value found. Please make sure all predictor variables are numeric."
                )

        numeric_rows.append(numeric_row)
        row_lengths.append(len(numeric_row))

    if len(set(row_lengths)) > 1:
        raise ValueError(
            "Missing value/s detected. Please ensure all rows have the same number of columns."
        )

    X = np.array(numeric_rows, dtype=float)

    if X.shape[1] < 1:
        raise ValueError("At least one predictor variable is required.")

    if headers:
        if use_first_column_as_id:
            headers = headers[1:]

        if require_class and len(headers) > X.shape[1]:
            headers = headers[:-1]

        feature_names = headers
    else:
        feature_names = [f"X{i + 1}" for i in range(X.shape[1])]

    return X, classes, sample_ids, feature_names

def classification_summary(y_true, y_pred):
    total = len(y_true)
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))

    return {
        "total_samples": total,
        "correct_classifications": correct,
        "overall_accuracy": f"{(correct / total) * 100:.2f}" if total else "0.00",
    }

def classification_rows(y_true, y_pred, probabilities, sample_ids, classes):
    rows = []

    for i, sample_id in enumerate(sample_ids):
        rows.append({
            "sample_id": str(sample_id),
            "true_class": str(y_true[i]) if y_true is not None else "",
            "predicted_class": str(y_pred[i]),
            "correct": "Yes" if y_true is not None and y_true[i] == y_pred[i] else "No" if y_true is not None else "",
            "probabilities": [
                f"{probabilities[i][j] * 100:.2f}"
                for j in range(len(classes))
            ],
        })

    return rows

def performance_by_class(y_true, y_pred, classes):
    rows = []

    for cls in classes:
        tp = sum((yt == cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == cls and yp != cls) for yt, yp in zip(y_true, y_pred))
        tn = sum((yt != cls and yp != cls) for yt, yp in zip(y_true, y_pred))

        sensitivity = tp / (tp + fn) * 100 if (tp + fn) else 0
        specificity = tn / (tn + fp) * 100 if (tn + fp) else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) else 0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity)
            else 0
        )

        rows.append({
            "class": cls,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "sensitivity": f"{sensitivity:.2f}",
            "specificity": f"{specificity:.2f}",
            "precision": f"{precision:.2f}",
            "f1": f"{f1:.2f}",
        })

    return rows

def make_confusion_plot(y_true, y_pred, classes, title):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=classes
    )

    fig = px.imshow(
        cm,
        x=classes,
        y=classes,
        text_auto=True,
        labels={
            "x": "Predicted Class",
            "y": "True Class",
            "color": "Count",
        },
        title=title
    )

    fig.update_layout(
        title_x=0.5,
        height=550
    )

    return fig.to_html(full_html=False)

def make_feature_importance(model, feature_names):
    importances = model.feature_importances_

    rows = [
        {
            "variable": feature_names[i],
            "importance": f"{importances[i] * 100:.2f}%",
            "value": importances[i] * 100,
        }
        for i in range(len(feature_names))
    ]

    rows = sorted(
        rows,
        key=lambda row: row["value"],
        reverse=True
    )

    fig = px.bar(
        x=[row["variable"] for row in rows],
        y=[row["value"] for row in rows],
        title="Feature Importance",
        labels={
            "x": "Variable",
            "y": "Importance (%)",
        }
    )

    fig.update_layout(
        title_x=0.5,
        height=550
    )

    table = [
        {
            "variable": row["variable"],
            "importance": row["importance"],
        }
        for row in rows
    ]

    return table, fig.to_html(full_html=False)

def make_tree_plot(model, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(16, 9))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[str(cls) for cls in class_names],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )

    buffer = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    image_base64 = base64.b64encode(image_png).decode("utf-8")

    return f"""
    <img src="data:image/png;base64,{image_base64}"
         style="max-width:100%; height:auto;">
    """

def build_tree_model(
    method,
    tree_max_depth=3,
    tree_criterion="gini",
    rf_n_estimators=100,
    rf_max_depth=5,
    rf_criterion="gini",
    xgb_n_estimators=100,
    xgb_learning_rate=0.1,
    xgb_max_depth=3
):
    if method == "decision_tree":
        return DecisionTreeClassifier(
            criterion=tree_criterion,
            max_depth=tree_max_depth,
            random_state=42
        )

    if method == "random_forest":
        return RandomForestClassifier(
            n_estimators=rf_n_estimators,
            criterion=rf_criterion,
            max_depth=rf_max_depth,
            random_state=42
        )

    if method == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ValueError(
                "XGBoost is not installed on the server. Please install xgboost before using this method."
            )

        return XGBClassifier(
            n_estimators=xgb_n_estimators,
            learning_rate=xgb_learning_rate,
            max_depth=xgb_max_depth,
            random_state=42,
            eval_metric="mlogloss"
        )

    raise ValueError("Invalid tree model selected.")

def encode_labels(labels, classes):
    label_map = {
        cls: i
        for i, cls in enumerate(classes)
    }

    inverse_label_map = {
        i: cls
        for cls, i in label_map.items()
    }

    encoded = np.array([
        label_map[label]
        for label in labels
    ])

    return encoded, label_map, inverse_label_map

def tree_models(request):

    active_tab = request.GET.get(
        "tab",
        "decision_tree"
    )

    method = active_tab

    data_key = f"data_tree_models_{method}"
    external_key = f"data_tree_models_external_{method}"
    results_key = f"results_tree_models_{method}"

    header_key = f"use_first_row_as_header_tree_models_{method}"
    id_key = f"use_first_column_as_id_tree_models_{method}"
    external_has_class_key = f"external_has_class_tree_models_{method}"

    normalization_key = f"normalization_tree_models_{method}"
    validation_key = f"tree_validation_mode_{method}"

    tree_max_depth_key = f"tree_max_depth_{method}"
    tree_criterion_key = f"tree_criterion_{method}"

    context = {
        "segment": "tree_models",
        "active_tab": active_tab,

        "data_tree_models": request.session.get(data_key, ""),
        "data_tree_models_external": request.session.get(external_key, ""),

        "use_first_row_as_header_tree_models": "checked" if request.session.get(header_key, False) else "",
        "use_first_column_as_id_tree_models": "checked" if request.session.get(id_key, False) else "",
        "external_has_class_tree_models": "checked" if request.session.get(external_has_class_key, False) else "",

        "normalization_tree_models": request.session.get(normalization_key, "none"),
        "tree_validation_mode": request.session.get(validation_key, "training"),

        "tree_max_depth": request.session.get(tree_max_depth_key, 3),
        "tree_criterion": request.session.get(tree_criterion_key, "gini"),

        "rf_n_estimators": 100,
        "rf_max_depth": 5,
        "rf_criterion": "gini",

        "xgb_n_estimators": 100,
        "xgb_learning_rate": 0.1,
        "xgb_max_depth": 3,

        "results_tree_models": request.session.get(results_key, None),
        "error_tree_models": None,
    }

    if request.method == "POST" and request.POST.get("clear_tree_models") == "true":
        for key in [
            data_key,
            external_key,
            results_key,
            header_key,
            id_key,
            external_has_class_key,
            normalization_key,
            validation_key,
            tree_max_depth_key,
            tree_criterion_key,
        ]:
            request.session.pop(key, None)

        context.update({
            "data_tree_models": "",
            "data_tree_models_external": "",
            "use_first_row_as_header_tree_models": "",
            "use_first_column_as_id_tree_models": "",
            "external_has_class_tree_models": "",
            "normalization_tree_models": "none",
            "tree_validation_mode": "training",
            "tree_max_depth": 3,
            "tree_criterion": "gini",
            "results_tree_models": None,
            "error_tree_models": None,
        })

        return render(
            request,
            "tree_models/tree_models.html",
            context
        )

    if request.method == "POST" and method in ["decision_tree", "random_forest", "xgboost"]:

        data = request.POST.get("data_tree_models", "")
        data_external = request.POST.get("data_tree_models_external", "")

        use_header = request.POST.get("use_first_row_as_header_tree_models") == "on"
        use_id = request.POST.get("use_first_column_as_id_tree_models") == "on"
        external_has_class = request.POST.get("external_has_class_tree_models") == "on"

        normalization = request.POST.get("normalization_tree_models", "none")
        validation_mode = request.POST.get("tree_validation_mode", "training")

        try:
            tree_max_depth = int(
                request.POST.get(
                    "tree_max_depth",
                    3
                )
            )
        except ValueError:
            tree_max_depth = 3

        tree_max_depth = max(
            1,
            min(
                50,
                tree_max_depth
            )
        )

        tree_criterion = request.POST.get(
            "tree_criterion",
            "gini"
        )

        if tree_criterion not in ["gini", "entropy"]:
            tree_criterion = "gini"

        try:
            rf_n_estimators = int(request.POST.get("rf_n_estimators", 100))
        except ValueError:
            rf_n_estimators = 100

        rf_n_estimators = max(10, min(1000, rf_n_estimators))

        try:
            rf_max_depth = int(request.POST.get("rf_max_depth", 5))
        except ValueError:
            rf_max_depth = 5

        rf_max_depth = max(1, min(50, rf_max_depth))

        rf_criterion = request.POST.get("rf_criterion", "gini")

        if rf_criterion not in ["gini", "entropy"]:
            rf_criterion = "gini"

        try:
            xgb_n_estimators = int(request.POST.get("xgb_n_estimators", 100))
        except ValueError:
            xgb_n_estimators = 100

        xgb_n_estimators = max(10, min(1000, xgb_n_estimators))

        try:
            xgb_learning_rate = float(request.POST.get("xgb_learning_rate", 0.1))
        except ValueError:
            xgb_learning_rate = 0.1

        xgb_learning_rate = max(0.001, min(1.0, xgb_learning_rate))

        try:
            xgb_max_depth = int(request.POST.get("xgb_max_depth", 3))
        except ValueError:
            xgb_max_depth = 3

        xgb_max_depth = max(1, min(20, xgb_max_depth))

        for key, value in {
            data_key: data,
            external_key: data_external,
            header_key: use_header,
            id_key: use_id,
            external_has_class_key: external_has_class,
            normalization_key: normalization,
            validation_key: validation_mode,
            tree_max_depth_key: tree_max_depth,
            tree_criterion_key: tree_criterion,
            f"rf_n_estimators_{method}": rf_n_estimators,
            f"rf_max_depth_{method}": rf_max_depth,
            f"rf_criterion_{method}": rf_criterion,
            f"xgb_n_estimators_{method}": xgb_n_estimators,
            f"xgb_learning_rate_{method}": xgb_learning_rate,
            f"xgb_max_depth_{method}": xgb_max_depth,
        }.items():
            request.session[key] = value

        context.update({
            "data_tree_models": data,
            "data_tree_models_external": data_external,
            "use_first_row_as_header_tree_models": "checked" if use_header else "",
            "use_first_column_as_id_tree_models": "checked" if use_id else "",
            "external_has_class_tree_models": "checked" if external_has_class else "",
            "normalization_tree_models": normalization,
            "tree_validation_mode": validation_mode,
            "tree_max_depth": tree_max_depth,
            "tree_criterion": tree_criterion,
            "rf_n_estimators": rf_n_estimators,
            "rf_max_depth": rf_max_depth,
            "rf_criterion": rf_criterion,
            "xgb_n_estimators": xgb_n_estimators,
            "xgb_learning_rate": xgb_learning_rate,
            "xgb_max_depth": xgb_max_depth,
        })

        try:
            

            X, y, sample_ids, feature_names = parse_tree_classification_data(
                data,
                use_header,
                use_id,
                require_class=True
            )

            X_scaled, scaler = normalize_data(
                X,
                normalization
            )

            classes = sorted(
                list(
                    set(y)
                )
            )

            class_counts = dict(
                Counter(y)
            )

            model = build_tree_model(
                method=method,
                tree_max_depth=tree_max_depth,
                tree_criterion=tree_criterion,
                rf_n_estimators=rf_n_estimators,
                rf_max_depth=rf_max_depth,
                rf_criterion=rf_criterion,
                xgb_n_estimators=xgb_n_estimators,
                xgb_learning_rate=xgb_learning_rate,
                xgb_max_depth=xgb_max_depth
            )

            label_map = {
                cls: i
                for i, cls in enumerate(classes)
            }

            inverse_label_map = {
                i: cls
                for cls, i in label_map.items()
            }

            if method == "xgboost":
                y_fit = np.array([
                    label_map[label]
                    for label in y
                ])
            else:
                y_fit = y

            model.fit(
                X_scaled,
                y_fit
            )

            y_pred = model.predict(
                X_scaled
            )

            if method == "xgboost":
                y_pred = np.array([
                    inverse_label_map[int(label)]
                    for label in y_pred
                ])

            probabilities = model.predict_proba(
                X_scaled
            )

            if method == "xgboost":
                probability_classes = classes
            else:
                probability_classes = [
                    str(cls)
                    for cls in model.classes_
                ]

            training_summary = classification_summary(
                y,
                y_pred
            )

            training_by_class = performance_by_class(
                y,
                y_pred,
                probability_classes
            )

            training_classification = classification_rows(
                y,
                y_pred,
                probabilities,
                sample_ids,
                probability_classes
            )

            confusion_plot = make_confusion_plot(
                y,
                y_pred,
                probability_classes,
                "Decision Tree - Training Confusion Matrix"
            )

            feature_importance_table, feature_importance_plot = make_feature_importance(
                model,
                feature_names
            )

            tree_plot = None
            if method == "decision_tree":
                tree_plot = make_tree_plot(
                    model,
                    feature_names,
                    probability_classes
                )

            if method == "decision_tree":
                model_parameters = [
                    {
                        "parameter": "Maximum Tree Depth",
                        "value": tree_max_depth,
                    },
                    {
                        "parameter": "Split Criterion",
                        "value": tree_criterion,
                    },
                ]
            elif method == "random_forest":
                model_parameters = [
                    {
                        "parameter": "Number of Trees",
                        "value": rf_n_estimators,
                    },
                    {
                        "parameter": "Maximum Tree Depth",
                        "value": rf_max_depth,
                    },
                    {
                        "parameter": "Split Criterion",
                        "value": rf_criterion,
                    },
                ]
            elif method == "xgboost":
                model_parameters = [
                    {
                        "parameter": "Boosting Rounds",
                        "value": xgb_n_estimators,
                    },
                    {
                        "parameter": "Learning Rate",
                        "value": xgb_learning_rate,
                    },
                    {
                        "parameter": "Maximum Tree Depth",
                        "value": xgb_max_depth,
                    },
                ]

            cv_summary = None
            cv_confusion_plot = None
            cv_by_class = None

            if validation_mode == "loo":
                loo = LeaveOneOut()

                cv_y_true = []
                cv_y_pred = []

                for train_idx, test_idx in loo.split(X_scaled):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train = np.array(y)[train_idx]
                    y_test = np.array(y)[test_idx]

                    cv_model = build_tree_model(
                        method=method,
                        tree_max_depth=tree_max_depth,
                        tree_criterion=tree_criterion,
                        rf_n_estimators=rf_n_estimators,
                        rf_max_depth=rf_max_depth,
                        rf_criterion=rf_criterion,
                        xgb_n_estimators=xgb_n_estimators,
                        xgb_learning_rate=xgb_learning_rate,
                        xgb_max_depth=xgb_max_depth
                    )

                    if method == "xgboost":
                        y_train_fit, _, inverse_label_map_cv = encode_labels(
                            y_train,
                            classes
                        )

                        cv_model.fit(
                            X_train,
                            y_train_fit
                        )

                        pred_encoded = cv_model.predict(X_test)

                        pred = np.array([
                            inverse_label_map_cv[int(label)]
                            for label in pred_encoded
                        ])

                    else:
                        cv_model.fit(
                            X_train,
                            y_train
                        )

                        pred = cv_model.predict(X_test)

                    cv_y_true.extend(y_test.tolist())
                    cv_y_pred.extend(pred.tolist())

                cv_summary = classification_summary(
                    cv_y_true,
                    cv_y_pred
                )

                cv_by_class = performance_by_class(
                    cv_y_true,
                    cv_y_pred,
                    probability_classes
                )

                cv_confusion_plot = make_confusion_plot(
                    cv_y_true,
                    cv_y_pred,
                    probability_classes,
                    "Decision Tree - Leave-One-Out Confusion Matrix"
                )

            split_summary = None
            split_confusion_plot = None
            split_by_class = None

            if validation_mode == "split":
                indices = np.arange(X_scaled.shape[0])

                train_idx, test_idx = train_test_split(
                    indices,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
                )

                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train = np.array(y)[train_idx]
                y_test = np.array(y)[test_idx]

                split_model = build_tree_model(
                    method=method,
                    tree_max_depth=tree_max_depth,
                    tree_criterion=tree_criterion,
                    rf_n_estimators=rf_n_estimators,
                    rf_max_depth=rf_max_depth,
                    rf_criterion=rf_criterion,
                    xgb_n_estimators=xgb_n_estimators,
                    xgb_learning_rate=xgb_learning_rate,
                    xgb_max_depth=xgb_max_depth
                )

                if method == "xgboost":
                    y_train_fit, _, inverse_label_map_split = encode_labels(
                        y_train,
                        classes
                    )

                    split_model.fit(
                        X_train,
                        y_train_fit
                    )

                    y_split_pred_encoded = split_model.predict(X_test)

                    y_split_pred = np.array([
                        inverse_label_map_split[int(label)]
                        for label in y_split_pred_encoded
                    ])

                else:
                    split_model.fit(
                        X_train,
                        y_train
                    )

                    y_split_pred = split_model.predict(X_test)

                split_summary = classification_summary(
                    y_test,
                    y_split_pred
                )

                split_summary["training_samples"] = len(train_idx)
                split_summary["validation_samples"] = len(test_idx)

                split_by_class = performance_by_class(
                    y_test,
                    y_split_pred,
                    probability_classes
                )

                split_confusion_plot = make_confusion_plot(
                    y_test,
                    y_split_pred,
                    probability_classes,
                    "Decision Tree - Random Split Confusion Matrix"
                )

            external_summary = None
            external_confusion_plot = None
            external_classification = None
            external_by_class = None

            if data_external.strip():
                X_ext, y_ext, ext_ids, _ = parse_tree_classification_data(
                    data_external,
                    use_header,
                    use_id,
                    require_class=external_has_class
                )

                if X_ext.shape[1] != X.shape[1]:
                    raise ValueError(
                        "External validation data must have the same number of variables as the training data."
                    )

                X_ext_scaled = scaler.transform(X_ext) if scaler is not None else X_ext

                y_ext_pred_raw = model.predict(X_ext_scaled)

                if method == "xgboost":
                    y_ext_pred = np.array([
                        inverse_label_map[int(label)]
                        for label in y_ext_pred_raw
                    ])
                else:
                    y_ext_pred = y_ext_pred_raw

                ext_probabilities = model.predict_proba(X_ext_scaled)

                if external_has_class:
                    external_summary = classification_summary(
                        y_ext,
                        y_ext_pred
                    )

                    external_by_class = performance_by_class(
                        y_ext,
                        y_ext_pred,
                        probability_classes
                    )

                    external_confusion_plot = make_confusion_plot(
                        y_ext,
                        y_ext_pred,
                        probability_classes,
                        "Decision Tree - External Validation Confusion Matrix"
                    )

                    external_classification = classification_rows(
                        y_ext,
                        y_ext_pred,
                        ext_probabilities,
                        ext_ids,
                        probability_classes
                    )

                else:
                    external_classification = classification_rows(
                        None,
                        y_ext_pred,
                        ext_probabilities,
                        ext_ids,
                        probability_classes
                    )

            results = {
                "method": {
                    "decision_tree": "Decision Tree",
                    "random_forest": "Random Forest",
                    "xgboost": "XGBoost",
                }[method],
                "validation_mode": validation_mode,
                "normalization": normalization,

                "model_parameters": model_parameters,

                "num_samples": len(sample_ids),
                "num_variables": X.shape[1],
                "num_classes": len(classes),
                "classes": classes,
                "class_counts": class_counts,

                "probability_classes": probability_classes,

                "tree_plot": tree_plot,

                "feature_importance_table": feature_importance_table,
                "feature_importance_plot": feature_importance_plot,

                "confusion_plot": confusion_plot,

                "training_summary": training_summary,
                "training_by_class": training_by_class,
                "training_classification": training_classification,

                "cv_summary": cv_summary,
                "cv_confusion_plot": cv_confusion_plot,
                "cv_by_class": cv_by_class,

                "split_summary": split_summary,
                "split_confusion_plot": split_confusion_plot,
                "split_by_class": split_by_class,

                "external_summary": external_summary,
                "external_confusion_plot": external_confusion_plot,
                "external_classification": external_classification,
                "external_by_class": external_by_class,
            }

            context["results_tree_models"] = results
            request.session[results_key] = results

        except Exception as e:
            context["error_tree_models"] = str(e)
            context["results_tree_models"] = None
            request.session[results_key] = None

    return render(
        request,
        "tree_models/tree_models.html",
        context
    )