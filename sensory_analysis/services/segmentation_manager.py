from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MINIMUM_CONSUMERS = 10
EXPLORATORY_CONSUMERS = 30
RANDOM_STATE = 42


def get_mapped_columns(mapping: dict, *, dataset_name: str, role: str) -> list[str]:
    dataset_mapping = mapping.get(dataset_name, {}) if isinstance(mapping, dict) else {}
    columns: list[str] = []
    for column_name, metadata in dataset_mapping.items():
        if isinstance(metadata, dict) and metadata.get("role") == role:
            columns.append(column_name)
        elif metadata == role:
            columns.append(column_name)
    return columns


def _find_column(
    dataframe: pd.DataFrame | None,
    mapping: dict,
    dataset_name: str,
    role: str,
    aliases: tuple[str, ...] = (),
) -> str | None:
    if dataframe is None or dataframe.empty:
        return None
    mapped = get_mapped_columns(mapping, dataset_name=dataset_name, role=role)
    if mapped and mapped[0] in dataframe.columns:
        return mapped[0]
    normalized = {str(col).strip().casefold(): col for col in dataframe.columns}
    for alias in aliases:
        match = normalized.get(alias.strip().casefold())
        if match is not None:
            return match
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _normalize_id(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})
    )


def _mode_or_none(series: pd.Series) -> Any:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return None
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def _display_name(feature: str) -> str:
    if feature.startswith("liking__"):
        return f"Liking — Product {feature.split('__', 1)[1]}"
    if feature.startswith("purchase__"):
        return f"Purchase intention — Product {feature.split('__', 1)[1]}"
    return feature.replace("_", " ").strip().title()


def prepare_segmentation_data(
    *,
    datasets: dict[str, pd.DataFrame],
    mapping: dict,
    include_purchase: bool = True,
) -> dict[str, Any]:
    evaluations = datasets.get("evaluations")
    if evaluations is None or evaluations.empty:
        return {"is_valid": False, "error": "The sensory evaluations dataset is empty."}

    consumer_col = _find_column(
        evaluations, mapping, "evaluations", "consumer_id", ("Consumer", "Consumer_ID", "Consumer ID")
    )
    sample_col = _find_column(
        evaluations, mapping, "evaluations", "sample_id", ("Sample", "Product", "Sample_ID", "Sample ID")
    )
    liking_col = _find_column(
        evaluations, mapping, "evaluations", "liking", ("Liking", "Overall_Liking", "Overall liking")
    )
    purchase_col = _find_column(
        evaluations,
        mapping,
        "evaluations",
        "purchase_intention",
        ("Purchase_Intention", "Purchase intention", "Purchase"),
    )

    missing = []
    if not consumer_col:
        missing.append("Consumer ID")
    if not sample_col:
        missing.append("Sample ID")
    if not liking_col:
        missing.append("Overall liking")
    if missing:
        return {"is_valid": False, "error": f"Missing mapped variables: {', '.join(missing)}."}

    columns = [consumer_col, sample_col, liking_col]
    if include_purchase and purchase_col:
        columns.append(purchase_col)
    data = evaluations[columns].copy()
    rename = {consumer_col: "consumer", sample_col: "product", liking_col: "liking"}
    if include_purchase and purchase_col:
        rename[purchase_col] = "purchase"
    data = data.rename(columns=rename)
    data["consumer"] = _normalize_id(data["consumer"])
    data["product"] = _normalize_id(data["product"])
    data["liking"] = _safe_numeric(data["liking"])
    if "purchase" in data.columns:
        data["purchase"] = _safe_numeric(data["purchase"])

    original_rows = len(data)
    data = data.dropna(subset=["consumer", "product", "liking"]).copy()
    excluded_rows = original_rows - len(data)
    duplicate_rows = int(data.duplicated(subset=["consumer", "product"], keep=False).sum())

    aggregation = {"liking": "mean"}
    if "purchase" in data.columns:
        aggregation["purchase"] = "mean"
    data = data.groupby(["consumer", "product"], as_index=False, observed=True).agg(aggregation)

    liking_matrix = data.pivot(index="consumer", columns="product", values="liking")
    liking_matrix.columns = [f"liking__{column}" for column in liking_matrix.columns]
    feature_frames = [liking_matrix]
    purchase_features: list[str] = []
    if "purchase" in data.columns:
        purchase_matrix = data.pivot(index="consumer", columns="product", values="purchase")
        purchase_matrix.columns = [f"purchase__{column}" for column in purchase_matrix.columns]
        feature_frames.append(purchase_matrix)
        purchase_features = purchase_matrix.columns.tolist()

    feature_matrix = pd.concat(feature_frames, axis=1).sort_index(axis=1)
    feature_matrix = feature_matrix.dropna(axis=1, how="all")
    if feature_matrix.shape[1] < 2:
        return {
            "is_valid": False,
            "error": "At least two usable preference variables are required for segmentation.",
        }

    consumers = feature_matrix.index.astype(str).tolist()
    consumer_profile = pd.DataFrame(index=feature_matrix.index)

    optional_sources = []
    for dataset_name in ("consumers", "general"):
        optional = datasets.get(dataset_name)
        if optional is None or optional.empty:
            continue
        optional_consumer = _find_column(
            optional, mapping, dataset_name, "consumer_id", ("Consumer", "Consumer_ID", "Consumer ID")
        )
        if not optional_consumer:
            continue
        optional_data = optional.copy()
        optional_data[optional_consumer] = _normalize_id(optional_data[optional_consumer])
        optional_data = optional_data.dropna(subset=[optional_consumer])
        optional_data = optional_data.drop_duplicates(subset=[optional_consumer], keep="first")
        optional_data = optional_data.set_index(optional_consumer)
        optional_data.index = optional_data.index.astype(str)
        optional_data = optional_data.drop(columns=[], errors="ignore")
        prefixed = optional_data.rename(columns={col: f"{dataset_name}__{col}" for col in optional_data.columns})
        consumer_profile = consumer_profile.join(prefixed, how="left")
        optional_sources.append(dataset_name)

    products = sorted(data["product"].astype(str).unique().tolist(), key=str.casefold)
    return {
        "is_valid": True,
        "data": data,
        "feature_matrix": feature_matrix,
        "consumer_profile": consumer_profile,
        "consumers": consumers,
        "products": products,
        "liking_features": liking_matrix.columns.tolist(),
        "purchase_features": purchase_features,
        "purchase_available": bool(purchase_features),
        "optional_sources": optional_sources,
        "excluded_rows": int(excluded_rows),
        "duplicate_rows": duplicate_rows,
        "consumer_count": int(feature_matrix.shape[0]),
        "feature_count": int(feature_matrix.shape[1]),
    }


def evaluate_cluster_solutions(
    scaled_data: np.ndarray,
    *,
    minimum_clusters: int = 2,
    maximum_clusters: int = 6,
) -> list[dict[str, Any]]:
    n_consumers = scaled_data.shape[0]
    maximum_allowed = min(maximum_clusters, n_consumers - 1)
    results: list[dict[str, Any]] = []
    for cluster_count in range(minimum_clusters, maximum_allowed + 1):
        try:
            model = KMeans(n_clusters=cluster_count, random_state=RANDOM_STATE, n_init=20)
            labels = model.fit_predict(scaled_data)
            counts = Counter(labels)
            if len(counts) < 2 or min(counts.values()) < 2:
                continue
            score = float(silhouette_score(scaled_data, labels))
            results.append(
                {
                    "cluster_count": cluster_count,
                    "silhouette_score": score,
                    "inertia": float(model.inertia_),
                    "minimum_segment_size": int(min(counts.values())),
                }
            )
        except (ValueError, FloatingPointError):
            continue
    return sorted(results, key=lambda item: (-item["silhouette_score"], item["cluster_count"]))


def _silhouette_label(score: float | None) -> str:
    if score is None:
        return "Not available"
    if score >= 0.70:
        return "Strong"
    if score >= 0.50:
        return "Reasonable"
    if score >= 0.25:
        return "Weak"
    return "Poor"


def _profile_optional_variables(profile: pd.DataFrame, labels: pd.Series) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if profile.empty:
        return results
    aligned = profile.copy()
    aligned["segment"] = labels
    for column in profile.columns:
        series = profile[column]
        numeric = _safe_numeric(series)
        numeric_ratio = float(numeric.notna().mean()) if len(numeric) else 0.0
        variable_name = column.split("__", 1)[-1]
        dataset_name = column.split("__", 1)[0] if "__" in column else "optional"
        segment_values = []
        for segment, group in aligned.groupby("segment", observed=True):
            original = group[column]
            if numeric_ratio >= 0.8:
                values = _safe_numeric(original).dropna()
                segment_values.append(
                    {
                        "segment": int(segment),
                        "value": float(values.mean()) if not values.empty else None,
                        "display": f"{values.mean():.2f}" if not values.empty else "—",
                        "n": int(values.size),
                    }
                )
            else:
                mode = _mode_or_none(original)
                valid = original.dropna().astype(str).str.strip()
                segment_values.append(
                    {
                        "segment": int(segment),
                        "value": mode,
                        "display": str(mode) if mode is not None else "—",
                        "n": int((valid != "").sum()),
                    }
                )
        results.append(
            {
                "dataset": dataset_name.title(),
                "variable": variable_name,
                "type": "numeric" if numeric_ratio >= 0.8 else "categorical",
                "segments": segment_values,
            }
        )
    return results


def run_segmentation_analysis(
    *,
    datasets: dict[str, pd.DataFrame],
    mapping: dict,
    configuration: dict | None = None,
    include_purchase: bool = True,
    requested_clusters: int | None = None,
    maximum_clusters: int = 6,
) -> dict[str, Any]:
    prepared = prepare_segmentation_data(
        datasets=datasets,
        mapping=mapping,
        include_purchase=include_purchase,
    )
    if not prepared.get("is_valid"):
        return prepared

    feature_matrix: pd.DataFrame = prepared["feature_matrix"]
    consumer_count = prepared["consumer_count"]
    if consumer_count < MINIMUM_CONSUMERS:
        return {
            "is_valid": False,
            "error": (
                f"At least {MINIMUM_CONSUMERS} consumers are required to run exploratory segmentation. "
                f"The current dataset contains {consumer_count}."
            ),
            "minimum_consumers": MINIMUM_CONSUMERS,
            "consumer_count": consumer_count,
        }

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(feature_matrix)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    solutions = evaluate_cluster_solutions(
        scaled,
        maximum_clusters=max(2, min(int(maximum_clusters), 6)),
    )
    if not solutions:
        return {
            "is_valid": False,
            "error": "No stable cluster solution could be calculated. Review missing values or increase the sample size.",
        }

    available_counts = {item["cluster_count"] for item in solutions}
    if requested_clusters in available_counts:
        selected_count = int(requested_clusters)
        selection_method = "User-selected"
    else:
        selected_count = int(solutions[0]["cluster_count"])
        selection_method = "Highest silhouette score"

    model = KMeans(n_clusters=selected_count, random_state=RANDOM_STATE, n_init=50)
    raw_labels = model.fit_predict(scaled)

    # Reorder segments by size, then by overall liking, for stable readable labels.
    overall_liking_columns = [i for i, col in enumerate(feature_matrix.columns) if col.startswith("liking__")]
    raw_summary = []
    for raw_segment in sorted(set(raw_labels)):
        mask = raw_labels == raw_segment
        size = int(mask.sum())
        overall_liking = float(np.nanmean(imputed[mask][:, overall_liking_columns])) if overall_liking_columns else 0.0
        raw_summary.append((raw_segment, size, overall_liking))
    raw_summary.sort(key=lambda item: (-item[1], -item[2], item[0]))
    label_map = {raw: index + 1 for index, (raw, _, _) in enumerate(raw_summary)}
    labels = pd.Series([label_map[value] for value in raw_labels], index=feature_matrix.index, name="segment")

    selected_solution = next(item for item in solutions if item["cluster_count"] == selected_count)
    silhouette = selected_solution["silhouette_score"]

    pca_components = min(2, scaled.shape[1], scaled.shape[0])
    pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(scaled)
    if pca_components == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(len(coordinates))])
        explained = [float(pca.explained_variance_ratio_[0] * 100), 0.0]
    else:
        explained = [float(value * 100) for value in pca.explained_variance_ratio_[:2]]

    consumer_assignments = []
    for index, consumer in enumerate(feature_matrix.index.astype(str)):
        consumer_assignments.append(
            {
                "consumer": consumer,
                "segment": int(labels.iloc[index]),
                "x": float(coordinates[index, 0]),
                "y": float(coordinates[index, 1]),
            }
        )

    original_with_segment = feature_matrix.copy()
    original_with_segment["segment"] = labels
    standardized_frame = pd.DataFrame(scaled, index=feature_matrix.index, columns=feature_matrix.columns)
    standardized_frame["segment"] = labels

    segment_profiles = []
    segment_sizes = labels.value_counts().sort_index()
    for segment in sorted(labels.unique()):
        segment_mask = labels == segment
        segment_features = original_with_segment.loc[segment_mask, feature_matrix.columns]
        standardized_means = standardized_frame.loc[segment_mask, feature_matrix.columns].mean()
        liking_means = {
            column.split("__", 1)[1]: float(segment_features[column].mean())
            for column in prepared["liking_features"]
            if column in segment_features
        }
        purchase_means = {
            column.split("__", 1)[1]: float(segment_features[column].mean())
            for column in prepared["purchase_features"]
            if column in segment_features
        }
        preferred_product = max(liking_means, key=liking_means.get) if liking_means else None
        top_standardized = standardized_means.sort_values(ascending=False).head(3)
        size = int(segment_sizes.loc[segment])
        label = f"Product {preferred_product} enthusiasts" if preferred_product else f"Segment {segment}"
        segment_profiles.append(
            {
                "segment": int(segment),
                "label": label,
                "consumer_count": size,
                "share": float(size / consumer_count * 100),
                "preferred_product": preferred_product,
                "overall_mean_liking": float(np.nanmean(list(liking_means.values()))) if liking_means else None,
                "overall_mean_purchase": float(np.nanmean(list(purchase_means.values()))) if purchase_means else None,
                "liking_means": liking_means,
                "purchase_means": purchase_means,
                "characteristics": [
                    {
                        "feature": feature,
                        "label": _display_name(feature),
                        "standardized_mean": float(value),
                    }
                    for feature, value in top_standardized.items()
                ],
            }
        )

    between_variance = standardized_frame.groupby("segment", observed=True)[feature_matrix.columns].mean().var(axis=0)
    differentiators = []
    for feature, variance in between_variance.sort_values(ascending=False).head(8).items():
        means = standardized_frame.groupby("segment", observed=True)[feature].mean()
        highest_segment = int(means.idxmax())
        lowest_segment = int(means.idxmin())
        differentiators.append(
            {
                "feature": feature,
                "label": _display_name(feature),
                "importance": float(variance),
                "highest_segment": highest_segment,
                "lowest_segment": lowest_segment,
                "range": float(means.max() - means.min()),
            }
        )

    optional_profiles = _profile_optional_variables(prepared["consumer_profile"], labels)

    liking_heatmap_rows = []
    for product in prepared["products"]:
        cells = []
        for segment in sorted(labels.unique()):
            profile = next(item for item in segment_profiles if item["segment"] == segment)
            cells.append(
                {
                    "segment": int(segment),
                    "value": profile["liking_means"].get(product),
                }
            )
        liking_heatmap_rows.append({"product": product, "cells": cells})

    warnings = []
    if consumer_count < EXPLORATORY_CONSUMERS:
        warnings.append(
            "Results are exploratory because fewer than 30 consumers were included. Stable segmentation generally requires a larger sample."
        )
    if silhouette < 0.25:
        warnings.append("The selected solution has poor cluster separation and should be interpreted cautiously.")
    elif silhouette < 0.50:
        warnings.append("The selected solution has limited cluster separation and should be treated as exploratory.")
    if feature_matrix.isna().any().any():
        warnings.append("Missing preference values were imputed using the median of each variable before clustering.")

    findings = []
    largest = max(segment_profiles, key=lambda item: item["consumer_count"])
    findings.append(
        f"{selected_count} consumer segments were identified using {selection_method.lower()} (silhouette = {silhouette:.3f})."
    )
    findings.append(
        f"Segment {largest['segment']} was the largest group, representing {largest['share']:.1f}% of included consumers."
    )
    for profile in segment_profiles:
        preferred = profile.get("preferred_product")
        if preferred:
            findings.append(
                f"Segment {profile['segment']} showed its highest mean liking for Product {preferred}."
            )
    if differentiators:
        top_labels = ", ".join(item["label"] for item in differentiators[:3])
        findings.append(f"The main variables differentiating the segments were {top_labels}.")

    chart_data = {
        "pca": {
            "points": consumer_assignments,
            "dimension_1_percentage": explained[0],
            "dimension_2_percentage": explained[1],
            "displayed_variance": explained[0] + explained[1],
        },
        "sizes": {
            "labels": [f"Segment {item['segment']}" for item in segment_profiles],
            "data": [item["consumer_count"] for item in segment_profiles],
        },
        "profiles": {
            "products": prepared["products"],
            "datasets": [
                {
                    "label": f"Segment {profile['segment']}",
                    "data": [profile["liking_means"].get(product) for product in prepared["products"]],
                }
                for profile in segment_profiles
            ],
        },
        "solutions": {
            "labels": [item["cluster_count"] for item in sorted(solutions, key=lambda item: item["cluster_count"])],
            "data": [item["silhouette_score"] for item in sorted(solutions, key=lambda item: item["cluster_count"])],
        },
    }

    return {
        "is_valid": True,
        "consumer_count": consumer_count,
        "feature_count": prepared["feature_count"],
        "segment_count": selected_count,
        "silhouette_score": silhouette,
        "silhouette_label": _silhouette_label(silhouette),
        "selection_method": selection_method,
        "include_purchase": bool(include_purchase and prepared["purchase_available"]),
        "purchase_available": prepared["purchase_available"],
        "products": prepared["products"],
        "features": [
            {"name": feature, "label": _display_name(feature)} for feature in feature_matrix.columns
        ],
        "solutions": sorted(solutions, key=lambda item: item["cluster_count"]),
        "segment_profiles": segment_profiles,
        "consumer_assignments": consumer_assignments,
        "differentiators": differentiators,
        "optional_profiles": optional_profiles,
        "liking_heatmap_rows": liking_heatmap_rows,
        "chart_data": chart_data,
        "warnings": warnings,
        "findings": findings,
        "data_quality": {
            "valid_consumers": consumer_count,
            "excluded_evaluation_rows": prepared["excluded_rows"],
            "duplicate_evaluation_rows": prepared["duplicate_rows"],
            "missing_values_imputed": int(feature_matrix.isna().sum().sum()),
            "optional_sources": prepared["optional_sources"],
        },
        "minimum_consumers": MINIMUM_CONSUMERS,
        "exploratory_threshold": EXPLORATORY_CONSUMERS,
    }