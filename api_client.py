import os
from typing import Any

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = (3, 10)


class ApiError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


def _error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, str) else ""


def _send(method: str, path: str, payload: dict | None = None) -> Any:
    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout as exc:
        raise ApiError("timeout", f"The backend at {API_BASE_URL} did not respond in time.") from exc
    except requests.ConnectionError as exc:
        raise ApiError("connection", f"Could not connect to the backend at {API_BASE_URL}.") from exc
    except requests.RequestException as exc:
        raise ApiError("connection", "The request to the backend failed.") from exc

    if response.status_code == 404:
        raise ApiError("not_found", f"{path} is not available on this backend.")
    if response.status_code == 422:
        raise ApiError("invalid_input", "The backend rejected one or more input values.")
    if response.status_code >= 400:
        message = f"The backend returned HTTP {response.status_code}."
        detail = _error_detail(response)
        raise ApiError("http", f"{message} {detail}".strip())

    try:
        return response.json()
    except ValueError as exc:
        raise ApiError("invalid_response", f"{path} did not return JSON.") from exc


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate(condition: bool, path: str) -> None:
    if not condition:
        raise ApiError("invalid_response", f"{path} returned an unexpected response format.")


def get_health() -> dict:
    data = _send("GET", "/health")
    _validate(isinstance(data, dict) and isinstance(data.get("model_loaded"), bool), "/health")
    return data


def get_service_info() -> dict:
    data = _send("GET", "/")
    _validate(isinstance(data, dict) and _is_number(data.get("threshold")), "/")
    return data


def predict(record: dict) -> dict:
    data = _send("POST", "/predict", record)
    _validate(
        isinstance(data, dict)
        and data.get("prediction") in (0, 1)
        and isinstance(data.get("label"), str)
        and _is_number(data.get("churn_probability"))
        and 0 <= data["churn_probability"] <= 1
        and _is_number(data.get("threshold")),
        "/predict",
    )
    return data


def get_model_info() -> dict:
    data = _send("GET", "/model-info")
    _validate(
        isinstance(data, dict)
        and isinstance(data.get("feature_columns"), list)
        and isinstance(data.get("test_metrics"), dict)
        and _is_number(data.get("threshold"))
        and _is_number(data.get("cv_f1")),
        "/model-info",
    )
    return data


def predict_batch(records: list[dict]) -> list[dict]:
    data = _send("POST", "/predict/batch", records)
    predictions = data.get("predictions") if isinstance(data, dict) else None
    _validate(
        isinstance(predictions, list)
        and len(predictions) == len(records)
        and all(
            isinstance(item, dict)
            and item.get("prediction") in (0, 1)
            and _is_number(item.get("churn_probability"))
            for item in predictions
        ),
        "/predict/batch",
    )
    return predictions
