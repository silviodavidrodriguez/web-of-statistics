import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
import pandas as pd
from django.core.cache import cache

CACHE_TIMEOUT = 60 * 60 * 4
CACHE_PREFIX = "sensory_study"
STATUS_NEW = "new"
STATUS_DATASETS_LOADED = "datasets_loaded"
STATUS_VARIABLES_MAPPED = "variables_mapped"
STATUS_CONFIGURED = "configured"
STATUS_VALIDATED = "validated"
STATUS_READY = "ready"

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def create_study_id() -> str:
    return uuid.uuid4().hex

def get_cache_key(study_id: str) -> str:
    return f"{CACHE_PREFIX}:{study_id}"

def dataframe_to_payload(dataframe: pd.DataFrame) -> dict[str, Any]:
    safe_dataframe = dataframe.copy()

    safe_dataframe = safe_dataframe.where(
        pd.notna(safe_dataframe),
        None,
    )

    return {
        "columns": list(safe_dataframe.columns),
        "records": safe_dataframe.to_dict(orient="records"),
    }

def payload_to_dataframe(
    payload: dict[str, Any] | None,
) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()

    return pd.DataFrame(
        payload.get("records", []),
        columns=payload.get("columns", []),
    )

def build_new_study(
    datasets: dict[str, pd.DataFrame] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    study_id = create_study_id()
    now = utc_now_iso()

    dataset_payloads = {
        dataset_name: dataframe_to_payload(dataframe)
        for dataset_name, dataframe in (datasets or {}).items()
    }

    return {
        "id": study_id,
        "metadata": {
            "name": "",
            "created_at": now,
            "updated_at": now,
            **(metadata or {}),
        },
        "datasets": dataset_payloads,
        "mapping": {},
        "configuration": {},
        "validation": {
            "errors": [],
            "warnings": [],
            "information": [],
        },
        "status": (
            STATUS_DATASETS_LOADED
            if dataset_payloads
            else STATUS_NEW
        ),
    }

def save_study(study: dict[str, Any]) -> None:
    study_copy = deepcopy(study)

    study_copy.setdefault("metadata", {})
    study_copy["metadata"]["updated_at"] = utc_now_iso()

    cache.set(
        get_cache_key(study_copy["id"]),
        study_copy,
        timeout=CACHE_TIMEOUT,
    )

def get_study(
    study_id: str | None,
) -> dict[str, Any] | None:
    if not study_id:
        return None

    study = cache.get(get_cache_key(study_id))

    if not study:
        return None

    return deepcopy(study)

def delete_study(study_id: str | None) -> None:
    if not study_id:
        return

    cache.delete(get_cache_key(study_id))

def get_study_datasets(
    study: dict[str, Any] | None,
) -> dict[str, pd.DataFrame]:
    if not study:
        return {}

    return {
        dataset_name: payload_to_dataframe(dataset_payload)
        for dataset_name, dataset_payload
        in study.get("datasets", {}).items()
    }

def replace_study_datasets(
    study: dict[str, Any],
    datasets: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    updated_study = deepcopy(study)

    updated_study["datasets"] = {
        dataset_name: dataframe_to_payload(dataframe)
        for dataset_name, dataframe in datasets.items()
    }

    updated_study["mapping"] = {}
    updated_study["configuration"] = {}
    updated_study["validation"] = {
        "errors": [],
        "warnings": [],
        "information": [],
    }
    updated_study["status"] = STATUS_DATASETS_LOADED

    return updated_study

def update_study_mapping(
    study: dict[str, Any],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    updated_study = deepcopy(study)
    updated_study["mapping"] = mapping
    updated_study["status"] = STATUS_VARIABLES_MAPPED

    return updated_study

def update_study_configuration(
    study: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    updated_study = deepcopy(study)
    updated_study["configuration"] = configuration
    updated_study["status"] = STATUS_CONFIGURED

    return updated_study

def update_study_validation(
    study: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    updated_study = deepcopy(study)
    updated_study["validation"] = validation

    if validation.get("errors"):
        updated_study["status"] = STATUS_VALIDATED
    else:
        updated_study["status"] = STATUS_READY

    return updated_study

def update_study_analysis(
    study: dict,
    *,
    analysis_name: str,
    result: dict,
) -> dict:
    analyses = study.setdefault(
        "analyses",
        {}
    )

    analyses[analysis_name] = result

    return study