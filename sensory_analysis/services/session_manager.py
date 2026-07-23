from __future__ import annotations

SESSION_PREFIX = "sensory_analysis_"

def clear_sensory_session(request) -> None:
    sensory_keys = [
        "sensory_study_id",
        "sensory_study_token",
        "sensory_dataset_names",
        "sensory_dataset_summary",
        "sensory_dataset_columns",
    ]

    for key in sensory_keys:
        request.session.pop(key, None)

    request.session.modified = True

def save_dataset_summary(
    request,
    summary: dict,
    columns: list[str],
) -> None:
    """
    Save only lightweight dataset metadata.

    The complete pasted table is deliberately not stored.
    """

    request.session[
        f"{SESSION_PREFIX}summary"
    ] = summary

    request.session[
        f"{SESSION_PREFIX}columns"
    ] = columns

def get_dataset_summary(request) -> dict | None:
    return request.session.get(
        f"{SESSION_PREFIX}summary"
    )

def get_dataset_columns(request) -> list[str]:
    return request.session.get(
        f"{SESSION_PREFIX}columns",
        [],
    )