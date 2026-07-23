from __future__ import annotations
import pandas as pd

MAX_ROWS = 100_000
MAX_COLUMNS = 300
MIN_ROWS = 2
MIN_COLUMNS = 3

class SensoryDataValidationError(Exception):
    """Raised when the dataset does not pass basic validation."""

def validate_dataset_structure(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Perform general structural checks before configuring sensory variables.
    """

    row_count, column_count = dataframe.shape

    errors: list[str] = []
    warnings: list[str] = []

    if row_count < MIN_ROWS:
        errors.append(
            "The dataset must contain at least two data rows."
        )

    if column_count < MIN_COLUMNS:
        errors.append(
            "The dataset must contain at least three columns."
        )

    if row_count > MAX_ROWS:
        errors.append(
            f"The dataset contains {row_count} rows. "
            f"The maximum allowed is {MAX_ROWS}."
        )

    if column_count > MAX_COLUMNS:
        errors.append(
            f"The dataset contains {column_count} columns. "
            f"The maximum allowed is {MAX_COLUMNS}."
        )

    unnamed_columns = [
        column
        for column in dataframe.columns
        if str(column).startswith("Unnamed")
    ]

    if unnamed_columns:
        warnings.append(
            "One or more columns do not have a descriptive name."
        )

    duplicated_rows = int(
        dataframe.duplicated().sum()
    )

    if duplicated_rows > 0:
        warnings.append(
            f"{duplicated_rows} completely duplicated rows were detected."
        )

    empty_columns = [
        str(column)
        for column in dataframe.columns
        if dataframe[column].replace("", pd.NA).isna().all()
    ]

    if empty_columns:
        warnings.append(
            "Completely empty columns were detected: "
            + ", ".join(empty_columns)
        )

    if errors:
        raise SensoryDataValidationError(
            " ".join(errors)
        )

    return {
        "row_count": int(row_count),
        "column_count": int(column_count),
        "duplicated_rows": duplicated_rows,
        "empty_columns": empty_columns,
        "warnings": warnings,
    }