from __future__ import annotations
import pandas as pd
from typing import Any

def build_dataset_preview(
    dataframe: pd.DataFrame,
    max_rows: int = 8,
    max_columns: int = 20,
) -> dict:
    """
    Create a small serializable preview for rendering in the template.

    The complete dataset is not included in the preview.
    """

    visible_columns = [
        str(column)
        for column in dataframe.columns[:max_columns]
    ]

    preview_dataframe = (
        dataframe
        .loc[:, visible_columns]
        .head(max_rows)
        .copy()
    )

    preview_dataframe = preview_dataframe.fillna("")

    rows = []

    for _, row in preview_dataframe.iterrows():
        rows.append(
            [
                safe_display_value(row[column])
                for column in visible_columns
            ]
        )

    missing_values = count_blank_values(dataframe)

    return {
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "visible_columns": visible_columns,
        "rows": rows,
        "missing_values": missing_values,
        "duplicated_rows": int(
            dataframe.duplicated().sum()
        ),
        "columns_truncated": (
            dataframe.shape[1] > max_columns
        ),
        "rows_truncated": (
            dataframe.shape[0] > max_rows
        ),
    }

def safe_display_value(value: Any) -> Any:
    if pd.isna(value):
        return ""

    return value

def count_blank_values(dataframe: pd.DataFrame) -> int:
    if dataframe.empty:
        return 0

    empty_strings = dataframe.map(
        lambda value: isinstance(value, str) and not value.strip()
    )

    return int(dataframe.isna().sum().sum() + empty_strings.sum().sum())

def build_dataset_preview(
    dataframe: pd.DataFrame,
    preview_rows: int = 8,
) -> dict[str, Any]:
    preview_dataframe = dataframe.head(preview_rows).copy()

    preview_records = []

    for _, row in preview_dataframe.iterrows():
        preview_records.append(
            [
                safe_display_value(row[column])
                for column in preview_dataframe.columns
            ]
        )

    return {
        "rows": int(len(dataframe)),
        "columns_count": int(len(dataframe.columns)),
        "blank_values": count_blank_values(dataframe),
        "duplicated_rows": int(dataframe.duplicated().sum()),
        "column_names": list(dataframe.columns),
        "preview_rows": preview_records,
        "preview_limit": preview_rows,
        "has_more_rows": len(dataframe) > preview_rows,
    }