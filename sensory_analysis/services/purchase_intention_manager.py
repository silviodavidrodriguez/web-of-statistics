from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon


ALPHA = 0.05


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
    normalized = {str(col).strip().lower(): col for col in dataframe.columns}
    for alias in aliases:
        if alias.strip().lower() in normalized:
            return normalized[alias.strip().lower()]
    return None


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _holm_adjust(p_values: list[float | None]) -> list[float | None]:
    adjusted: list[float | None] = [None] * len(p_values)
    valid = [(idx, float(p)) for idx, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    valid.sort(key=lambda item: item[1])
    running = 0.0
    m = len(valid)
    for rank, (idx, p_value) in enumerate(valid):
        candidate = min(1.0, (m - rank) * p_value)
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def _natural_join(values: list[str]) -> str:
    values = [str(v) for v in values if str(v).strip()]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _format_signed(value: float) -> str:
    return f"+{value:.2f}" if value > 0 else (f"−{abs(value):.2f}" if value < 0 else "0.00")


def prepare_purchase_data(
    *,
    datasets: dict,
    mapping: dict,
    purchase_variable: str | None = None,
) -> dict:
    evaluations = datasets.get("evaluations")
    if evaluations is None or evaluations.empty:
        return {"is_valid": False, "error": "The sensory evaluations dataset is empty."}

    consumer_col = _find_column(evaluations, mapping, "evaluations", "consumer_id", ("Consumer", "Consumer_ID"))
    sample_col = _find_column(evaluations, mapping, "evaluations", "sample_id", ("Sample", "Product", "Sample_ID"))
    purchase_columns = get_mapped_columns(mapping, dataset_name="evaluations", role="purchase_intention")
    if purchase_variable and purchase_variable in evaluations.columns:
        purchase_col = purchase_variable
    elif purchase_columns:
        purchase_col = purchase_columns[0]
    else:
        purchase_col = _find_column(evaluations, mapping, "evaluations", "purchase_intention", ("Purchase_Intention", "Purchase intention", "Purchase"))

    liking_col = _find_column(evaluations, mapping, "evaluations", "liking", ("Liking", "Overall_Liking", "Overall liking"))

    missing = []
    if not consumer_col:
        missing.append("Consumer ID")
    if not sample_col:
        missing.append("Sample ID")
    if not purchase_col:
        missing.append("Purchase intention")
    if missing:
        return {"is_valid": False, "error": f"Missing mapped variables: {', '.join(missing)}."}

    columns = [consumer_col, sample_col, purchase_col]
    if liking_col:
        columns.append(liking_col)
    data = evaluations[columns].copy()
    data = data.rename(columns={consumer_col: "consumer", sample_col: "product", purchase_col: "purchase"})
    if liking_col:
        data = data.rename(columns={liking_col: "liking"})
    data["purchase"] = pd.to_numeric(data["purchase"], errors="coerce")
    if "liking" in data.columns:
        data["liking"] = pd.to_numeric(data["liking"], errors="coerce")

    excluded_rows = int(data[["consumer", "product", "purchase"]].isna().any(axis=1).sum())
    data = data.dropna(subset=["consumer", "product", "purchase"])
    if data.empty:
        return {"is_valid": False, "error": "No valid purchase-intention evaluations were found."}

    duplicate_count = int(data.duplicated(subset=["consumer", "product"], keep=False).sum())
    aggregation = {"purchase": "mean"}
    if "liking" in data.columns:
        aggregation["liking"] = "mean"
    data = data.groupby(["consumer", "product"], as_index=False).agg(aggregation)

    observed = data["purchase"].dropna()
    observed_min = int(np.floor(observed.min()))
    observed_max = int(np.ceil(observed.max()))
    integer_like = bool(np.allclose(observed, np.round(observed)))
    if integer_like and observed_min >= 1 and observed_max <= 5:
        scale_min, scale_max = 1, 5
    else:
        scale_min, scale_max = observed_min, observed_max
    categories = list(range(scale_min, scale_max + 1)) if scale_max - scale_min <= 10 else sorted(observed.unique().tolist())
    favorable_min = scale_max - 1 if scale_max - scale_min >= 2 else scale_max
    unfavorable_max = scale_min + 1 if scale_max - scale_min >= 2 else scale_min

    return {
        "is_valid": True,
        "data": data,
        "purchase_variable": purchase_col,
        "liking_variable": liking_col,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "categories": categories,
        "favorable_min": favorable_min,
        "unfavorable_max": unfavorable_max,
        "excluded_rows": excluded_rows,
        "duplicate_rows": duplicate_count,
    }


def build_product_summaries(data: pd.DataFrame, categories: list, favorable_min: float, unfavorable_max: float) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    distributions: list[dict] = []
    for product, group in data.groupby("product", sort=True):
        values = group["purchase"].dropna()
        n = int(values.size)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n else 0.0
        ci = 1.96 * se
        favorable_n = int((values >= favorable_min).sum())
        unfavorable_n = int((values <= unfavorable_max).sum())
        neutral_n = n - favorable_n - unfavorable_n
        summary = {
            "product": str(product),
            "n": n,
            "mean": mean,
            "median": float(values.median()),
            "sd": sd,
            "ci_low": mean - ci,
            "ci_high": mean + ci,
            "favorable_n": favorable_n,
            "favorable_percentage": favorable_n / n * 100 if n else 0.0,
            "neutral_n": neutral_n,
            "neutral_percentage": neutral_n / n * 100 if n else 0.0,
            "unfavorable_n": unfavorable_n,
            "unfavorable_percentage": unfavorable_n / n * 100 if n else 0.0,
            "mean_display": f"{mean:.2f}",
            "ci_display": f"{mean - ci:.2f}–{mean + ci:.2f}",
        }
        summaries.append(summary)
        counts = values.value_counts().to_dict()
        distributions.append({
            "product": str(product),
            "categories": [
                {"category": str(category), "count": int(counts.get(category, 0)), "percentage": float(counts.get(category, 0) / n * 100 if n else 0.0)}
                for category in categories
            ],
        })
    summaries.sort(key=lambda row: (-row["mean"], -row["favorable_percentage"], row["product"]))
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
        row["rank_badge"] = medals.get(rank, f"#{rank}")
    return summaries, distributions


def run_friedman_and_pairwise(data: pd.DataFrame, alpha: float = ALPHA) -> dict:
    pivot = data.pivot(index="consumer", columns="product", values="purchase")
    complete = pivot.dropna(axis=0, how="any")
    products = [str(col) for col in complete.columns]
    result = {
        "calculable": False,
        "products": products,
        "complete_consumers": int(complete.shape[0]),
        "statistic": None,
        "df": max(0, len(products) - 1),
        "p_value": None,
        "significant": False,
        "kendall_w": None,
        "small_sample": complete.shape[0] < 10,
        "pairwise": [],
    }
    if complete.shape[0] < 2 or complete.shape[1] < 2:
        return result
    arrays = [complete[col].to_numpy(dtype=float) for col in complete.columns]
    try:
        statistic, p_value = friedmanchisquare(*arrays)
    except ValueError:
        return result
    n, k = complete.shape
    result.update({
        "calculable": True,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "kendall_w": float(statistic / (n * (k - 1))) if n and k > 1 else None,
    })

    if p_value < alpha:
        comparisons = []
        raw_p_values = []
        for col_a, col_b in combinations(complete.columns, 2):
            a = complete[col_a].to_numpy(dtype=float)
            b = complete[col_b].to_numpy(dtype=float)
            differences = a - b
            if np.allclose(differences, 0):
                stat, pair_p = 0.0, 1.0
            else:
                try:
                    stat, pair_p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="auto")
                except ValueError:
                    stat, pair_p = 0.0, 1.0
            comparisons.append({
                "product_a": str(col_a),
                "product_b": str(col_b),
                "comparison": f"{col_a} vs {col_b}",
                "paired_consumers": int(len(a)),
                "statistic": float(stat),
                "p_value": float(pair_p),
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "direction": (f"{col_a} > {col_b}" if np.mean(a) > np.mean(b) else (f"{col_b} > {col_a}" if np.mean(b) > np.mean(a) else "No directional difference")),
                "higher_product": (str(col_a) if np.mean(a) > np.mean(b) else (str(col_b) if np.mean(b) > np.mean(a) else None)),
                "lower_product": (str(col_b) if np.mean(a) > np.mean(b) else (str(col_a) if np.mean(b) > np.mean(a) else None)),
                "direction_detail": (
                    f"Product {col_a} had higher purchase intention" if np.mean(a) > np.mean(b)
                    else f"Product {col_b} had higher purchase intention" if np.mean(b) > np.mean(a)
                    else "No directional difference"
                ),
            })
            raw_p_values.append(float(pair_p))
        adjusted = _holm_adjust(raw_p_values)
        for row, adj in zip(comparisons, adjusted):
            row["adjusted_p_value"] = adj
            row["significant"] = bool(adj is not None and adj < alpha)
            row["result"] = "Significant" if row["significant"] else "Not significant"
        result["pairwise"] = comparisons
    return result


def analyze_liking_relationship(data: pd.DataFrame, alpha: float = ALPHA) -> dict:
    if "liking" not in data.columns:
        return {"available": False, "reason": "No overall liking variable was mapped."}
    valid = data.dropna(subset=["purchase", "liking"])
    if valid.shape[0] < 3:
        return {"available": False, "reason": "At least three paired liking and purchase-intention observations are required."}
    coefficient, p_value = spearmanr(valid["liking"], valid["purchase"])
    if not np.isfinite(coefficient) or not np.isfinite(p_value):
        return {"available": False, "reason": "The correlation could not be calculated because one variable has no variation."}
    abs_r = abs(float(coefficient))
    strength = "very weak" if abs_r < 0.2 else "weak" if abs_r < 0.4 else "moderate" if abs_r < 0.6 else "strong" if abs_r < 0.8 else "very strong"
    direction = "positive" if coefficient > 0 else "negative" if coefficient < 0 else "no"
    return {
        "available": True,
        "n": int(valid.shape[0]),
        "coefficient": float(coefficient),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "strength": strength,
        "direction": direction,
        "scatter_points": [
            {"x": float(row.liking), "y": float(row.purchase), "product": str(row.product), "consumer": str(row.consumer)}
            for row in valid.itertuples(index=False)
        ],
    }


def analyze_information_effect(datasets: dict, mapping: dict, purchase_data: pd.DataFrame, alpha: float = ALPHA) -> dict:
    general = datasets.get("general")
    if general is None or general.empty:
        return {"available": False, "reason": "No general study questions dataset was provided."}
    consumer_col = _find_column(general, mapping, "general", "consumer_id", ("Consumer", "Consumer_ID"))
    after_col = _find_column(general, mapping, "general", "purchase_with_information", ("Purchase_With_Information", "Purchase with information", "Purchase_After_Information"))
    functional_col = _find_column(general, mapping, "general", "functional_information", ("Functional_Information", "Functional information"))
    if not consumer_col or not after_col:
        return {"available": False, "reason": "Map Consumer ID and Purchase With Information in the general questions dataset to analyze information effects."}

    general_data = general[[consumer_col, after_col] + ([functional_col] if functional_col else [])].copy()
    general_data = general_data.rename(columns={consumer_col: "consumer", after_col: "purchase_after"})
    if functional_col:
        general_data = general_data.rename(columns={functional_col: "functional_information"})
    general_data["purchase_after"] = pd.to_numeric(general_data["purchase_after"], errors="coerce")
    baseline = purchase_data.groupby("consumer", as_index=False)["purchase"].mean().rename(columns={"purchase": "purchase_before"})
    merged = baseline.merge(general_data, on="consumer", how="inner").dropna(subset=["purchase_before", "purchase_after"])
    if merged.empty:
        return {"available": False, "reason": "No consumers could be matched between evaluations and general questions."}
    merged["uplift"] = merged["purchase_after"] - merged["purchase_before"]
    test_calculable = merged.shape[0] >= 2 and not np.allclose(merged["uplift"], 0)
    statistic = p_value = None
    if test_calculable:
        try:
            statistic, p_value = wilcoxon(merged["purchase_after"], merged["purchase_before"], method="auto")
            statistic, p_value = float(statistic), float(p_value)
        except ValueError:
            test_calculable = False
    groups = []
    if "functional_information" in merged.columns:
        for label, group in merged.groupby("functional_information", dropna=False):
            groups.append({
                "group": str(label),
                "n": int(group.shape[0]),
                "before_mean": float(group["purchase_before"].mean()),
                "after_mean": float(group["purchase_after"].mean()),
                "uplift_mean": float(group["uplift"].mean()),
            })
    return {
        "available": True,
        "n": int(merged.shape[0]),
        "before_mean": float(merged["purchase_before"].mean()),
        "after_mean": float(merged["purchase_after"].mean()),
        "uplift_mean": float(merged["uplift"].mean()),
        "uplift_display": _format_signed(float(merged["uplift"].mean())),
        "positive_uplift_percentage": float((merged["uplift"] > 0).mean() * 100),
        "test_calculable": test_calculable,
        "statistic": statistic,
        "p_value": p_value,
        "significant": bool(p_value is not None and p_value < alpha),
        "groups": groups,
        "note": "The baseline is each consumer's mean product-level purchase intention because the information response is recorded once per consumer, not once per product.",
    }


def build_distribution_summary(product_summaries: list[dict]) -> list[str]:
    statements: list[str] = []
    for row in product_summaries:
        product = row["product"]
        favorable = row["favorable_percentage"]
        neutral = row["neutral_percentage"]
        unfavorable = row["unfavorable_percentage"]
        if favorable >= 80:
            text = f"Product {product} concentrated predominantly favorable responses ({favorable:.1f}%)."
        elif unfavorable >= 50:
            text = f"Product {product} concentrated mainly unfavorable responses ({unfavorable:.1f}%)."
        elif neutral >= 30:
            text = f"Product {product} showed a mixed response pattern, including {neutral:.1f}% neutral responses."
        else:
            text = f"Product {product} recorded {favorable:.1f}% favorable, {neutral:.1f}% neutral, and {unfavorable:.1f}% unfavorable responses."
        statements.append(text)
    return statements


def build_executive_summary(product_summaries: list[dict], friedman: dict, liking: dict, information: dict) -> dict:
    top = product_summaries[0] if product_summaries else None
    return {
        "products_ranked": len(product_summaries),
        "top_product": top["product"] if top else "—",
        "highest_mean": top["mean"] if top else None,
        "overall_difference": (
            "Significant" if friedman.get("calculable") and friedman.get("significant")
            else "Not significant" if friedman.get("calculable")
            else "Not calculable"
        ),
        "overall_difference_exploratory": bool(
            friedman.get("calculable")
            and friedman.get("significant")
            and friedman.get("small_sample")
        ),
        "liking_association": (
            f"{liking.get('strength', '').capitalize()} {liking.get('direction', '')}" if liking.get("available")
            else "Not available"
        ),
        "mean_uplift": information.get("uplift_display", "—") if information.get("available") else "—",
    }


def build_chart_data(product_summaries: list[dict], distributions: list[dict], categories: list) -> dict:
    products = [row["product"] for row in product_summaries]
    distribution_lookup = {row["product"]: row for row in distributions}
    distribution_datasets = []
    for category in categories:
        distribution_datasets.append({
            "label": str(category),
            "data": [next((item["percentage"] for item in distribution_lookup[p]["categories"] if item["category"] == str(category)), 0.0) for p in products],
        })
    return {
        "products": products,
        "distribution": {"labels": products, "datasets": distribution_datasets},
        "favorable": {
            "labels": products,
            "data": [row["favorable_percentage"] for row in product_summaries],
            "means": [row["mean"] for row in product_summaries],
        },
    }


def build_interpretation(product_summaries: list[dict], friedman: dict, liking: dict, information: dict) -> list[str]:
    statements: list[str] = []
    if product_summaries:
        best = product_summaries[0]
        statements.append(
            f"Product {best['product']} achieved the highest mean purchase intention ({best['mean']:.2f}) and the highest favorable-response rate ({best['favorable_percentage']:.1f}%)."
        )
    if friedman.get("calculable"):
        if friedman.get("significant"):
            statements.append(
                f"Purchase intention differed significantly among products according to the Friedman test (p = {friedman['p_value']:.4f}, Kendall's W = {friedman['kendall_w']:.3f})."
            )
            significant_pairs = [row["comparison"] for row in friedman.get("pairwise", []) if row.get("significant")]
            if significant_pairs:
                statements.append(f"After Holm adjustment, significant paired differences remained for {_natural_join(significant_pairs)}.")
            else:
                statements.append("No individual product comparison remained significant after Holm adjustment.")
        else:
            statements.append(f"The Friedman test did not detect significant differences in purchase intention among products (p = {friedman['p_value']:.4f}).")
        if friedman.get("small_sample"):
            statements.append("Inferential results should be interpreted cautiously because fewer than 10 consumers had complete observations across all products.")
    if liking.get("available"):
        significance = "significant" if liking["significant"] else "not statistically significant"
        statements.append(
            f"Overall liking and purchase intention showed a {liking['strength']} {liking['direction']} Spearman association (ρ = {liking['coefficient']:.3f}, p = {liking['p_value']:.4f}), which was {significance}."
        )
    if information.get("available"):
        if information.get("test_calculable") and information.get("p_value") is not None:
            significance_text = (
                "and the difference was statistically significant"
                if information.get("significant")
                else "but the difference was not statistically significant"
            )
            statements.append(
                f"Purchase intention after information was {_format_signed(information['uplift_mean'])} points higher than the consumer-level baseline on average, {significance_text} (p = {information['p_value']:.4f}). This comparison cannot be attributed to a specific product with the current data structure."
            )
        else:
            statements.append(
                f"Purchase intention after information was {_format_signed(information['uplift_mean'])} points relative to each consumer's mean baseline purchase intention. This comparison is consumer-level and cannot be attributed to a specific product with the current data structure."
            )
    return statements


def run_purchase_intention_analysis(
    *,
    datasets: dict,
    mapping: dict,
    configuration: dict | None = None,
    purchase_variable: str | None = None,
) -> dict:
    prepared = prepare_purchase_data(datasets=datasets, mapping=mapping, purchase_variable=purchase_variable)
    if not prepared.get("is_valid"):
        return prepared
    data = prepared["data"]
    summaries, distributions = build_product_summaries(data, prepared["categories"], prepared["favorable_min"], prepared["unfavorable_max"])
    friedman = run_friedman_and_pairwise(data)
    liking = analyze_liking_relationship(data)
    information = analyze_information_effect(datasets, mapping, data)
    charts = build_chart_data(summaries, distributions, prepared["categories"])
    distribution_summary = build_distribution_summary(summaries)
    executive_summary = build_executive_summary(summaries, friedman, liking, information)
    return {
        "is_valid": True,
        "purchase_variable": prepared["purchase_variable"],
        "liking_variable": prepared["liking_variable"],
        "summary": {
            "products": int(data["product"].nunique()),
            "consumers": int(data["consumer"].nunique()),
            "valid_evaluations": int(data.shape[0]),
            "scale_min": prepared["scale_min"],
            "scale_max": prepared["scale_max"],
            "favorable_min": prepared["favorable_min"],
            "unfavorable_max": prepared["unfavorable_max"],
            "overall_mean": float(data["purchase"].mean()),
            "overall_median": float(data["purchase"].median()),
        },
        "product_summaries": summaries,
        "distributions": distributions,
        "distribution_summary": distribution_summary,
        "executive_summary": executive_summary,
        "friedman": friedman,
        "liking_relationship": liking,
        "information_effect": information,
        "chart_data": charts,
        "data_quality": {
            "excluded_rows": prepared["excluded_rows"],
            "duplicate_rows_consolidated": prepared["duplicate_rows"],
            "valid_evaluations": int(data.shape[0]),
        },
        "interpretation": build_interpretation(summaries, friedman, liking, information),
    }