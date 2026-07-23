from __future__ import annotations
from io import StringIO
import pandas as pd

class SensoryDataParseError(Exception):
    """Raised when pasted sensory data cannot be interpreted."""

def parse_pasted_table(
    pasted_data: str,
    has_header: bool = True,
) -> pd.DataFrame:
    """
    Convert a table copied from Excel into a pandas DataFrame.

    Excel normally places tab characters between columns and newline
    characters between rows when a range is copied.
    """

    cleaned_text = clean_pasted_text(pasted_data)

    if not cleaned_text:
        raise SensoryDataParseError(
            "No data were pasted."
        )

    header = 0 if has_header else None

    try:
        dataframe = pd.read_csv(
            StringIO(cleaned_text),
            sep="\t",
            header=header,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )

    except Exception as exc:
        raise SensoryDataParseError(
            "The pasted content could not be interpreted as a table. "
            "Copy the cells directly from Excel and paste them without "
            "modifying the separators."
        ) from exc

    if dataframe.empty:
        raise SensoryDataParseError(
            "The pasted table does not contain data rows."
        )

    if not has_header:
        dataframe.columns = [
            f"Column_{index + 1}"
            for index in range(dataframe.shape[1])
        ]

    dataframe = clean_dataframe(dataframe)

    return dataframe

def clean_pasted_text(pasted_data: str) -> str:
    """
    Normalize line endings and remove empty lines surrounding the table.
    """

    if not isinstance(pasted_data, str):
        return ""

    normalized = pasted_data.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    lines = normalized.split("\n")

    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)

def clean_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize column names and cell contents.
    """

    cleaned = dataframe.copy()

    cleaned.columns = make_unique_column_names(
        [
            normalize_column_name(column)
            for column in cleaned.columns
        ]
    )

    for column in cleaned.columns:
        cleaned[column] = (
            cleaned[column]
            .astype(str)
            .str.strip()
        )

    empty_rows = (
        cleaned
        .replace("", pd.NA)
        .isna()
        .all(axis=1)
    )

    cleaned = cleaned.loc[~empty_rows].copy()
    cleaned.reset_index(drop=True, inplace=True)

    return cleaned

def normalize_column_name(column) -> str:
    """
    Remove surrounding spaces and normalize unnamed columns.
    """

    column_name = str(column).strip()

    if not column_name or column_name.lower().startswith("unnamed"):
        return "Unnamed"

    return column_name

def make_unique_column_names(
    column_names: list[str],
) -> list[str]:
    """
    Ensure that duplicated column names receive a numerical suffix.
    """

    counts: dict[str, int] = {}
    unique_names: list[str] = []

    for name in column_names:
        base_name = name or "Unnamed"

        counts[base_name] = counts.get(base_name, 0) + 1

        if counts[base_name] == 1:
            unique_names.append(base_name)
        else:
            unique_names.append(
                f"{base_name}_{counts[base_name]}"
            )

    return unique_names