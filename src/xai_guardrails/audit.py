"""Binary classification quality, fairness, and calibration audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class AuditThresholds:
    """Maximum tolerated gaps/errors and minimum subgroup support."""

    max_demographic_parity_gap: float = 0.10
    max_equal_opportunity_gap: float = 0.10
    max_equalized_odds_gap: float = 0.10
    min_selection_rate_ratio: float = 0.80
    max_ece: float = 0.10
    min_group_size: int = 20


@dataclass
class AuditReport:
    """Serializable result of a binary classification audit."""

    sample_count: int
    threshold: float
    overall: dict[str, float | int | None]
    groups: dict[str, dict[str, float | int | None]]
    fairness: dict[str, float | None]
    findings: list[dict[str, Any]]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(np.digitize(y_score, edges[1:-1]), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            error += np.mean(mask) * abs(float(np.mean(y_score[mask])) - float(np.mean(y_true[mask])))
    return float(error)


def audit_predictions(
    y_true: Any,
    y_score: Any,
    sensitive_features: Any,
    *,
    threshold: float = 0.5,
    calibration_bins: int = 10,
    thresholds: AuditThresholds | None = None,
) -> AuditReport:
    """Evaluate binary labels and positive-class probabilities by sensitive group."""
    limits = thresholds or AuditThresholds()
    labels = _as_vector(y_true, "y_true")
    scores = _as_vector(y_score, "y_score").astype(float)
    groups = _as_vector(sensitive_features, "sensitive_features")
    if not (len(labels) == len(scores) == len(groups)):
        raise ValueError("y_true, y_score, and sensitive_features must have equal lengths")
    if set(np.unique(labels).tolist()) - {0, 1, False, True}:
        raise ValueError("y_true must contain only binary values 0 and 1")
    labels = labels.astype(int)
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("y_score must contain finite probabilities between 0 and 1")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be at least 1")
    if limits.min_group_size < 1:
        raise ValueError("min_group_size must be at least 1")

    predictions = (scores >= threshold).astype(int)
    overall: dict[str, float | int | None] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None,
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "ece": _expected_calibration_error(labels, scores, calibration_bins),
    }

    group_metrics: dict[str, dict[str, float | int | None]] = {}
    for group in np.unique(groups):
        mask = groups == group
        group_labels = labels[mask]
        group_predictions = predictions[mask]
        positive_count = int(np.sum(group_labels == 1))
        negative_count = int(np.sum(group_labels == 0))
        group_metrics[str(group)] = {
            "support": int(np.sum(mask)),
            "selection_rate": float(np.mean(group_predictions)),
            "accuracy": float(accuracy_score(group_labels, group_predictions)),
            "true_positive_rate": float(np.sum((group_labels == 1) & (group_predictions == 1)) / positive_count)
            if positive_count
            else None,
            "false_positive_rate": float(np.sum((group_labels == 0) & (group_predictions == 1)) / negative_count)
            if negative_count
            else None,
        }

    selection_rates = [float(metrics["selection_rate"]) for metrics in group_metrics.values()]
    tpr_values = [float(metrics["true_positive_rate"]) for metrics in group_metrics.values() if metrics["true_positive_rate"] is not None]
    fpr_values = [float(metrics["false_positive_rate"]) for metrics in group_metrics.values() if metrics["false_positive_rate"] is not None]
    has_missing_tpr = any(metrics["true_positive_rate"] is None for metrics in group_metrics.values())
    has_missing_fpr = any(metrics["false_positive_rate"] is None for metrics in group_metrics.values())
    dp_gap = max(selection_rates) - min(selection_rates) if len(selection_rates) > 1 else 0.0
    eo_gap = None if len(group_metrics) > 1 and (has_missing_tpr or has_missing_fpr) else (max(tpr_values) - min(tpr_values) if len(tpr_values) > 1 else 0.0)
    fpr_gap = None if len(group_metrics) > 1 and (has_missing_tpr or has_missing_fpr) else (max(fpr_values) - min(fpr_values) if len(fpr_values) > 1 else 0.0)
    eo_gap_for_eod = eo_gap if eo_gap is not None else 0.0
    fpr_gap_for_eod = fpr_gap if fpr_gap is not None else 0.0
    fairness = {
        "demographic_parity_gap": dp_gap,
        "equal_opportunity_gap": eo_gap,
        "equalized_odds_gap": max(eo_gap_for_eod, fpr_gap_for_eod) if len(group_metrics) > 1 else 0.0,
        "selection_rate_ratio": min(selection_rates) / max(selection_rates) if max(selection_rates) > 0 else 1.0,
    }

    findings: list[dict[str, Any]] = []

    def record(name: str, value: float | None, limit: float, direction: str, *, group: str | None = None) -> None:
        if value is None:
            findings.append({
                "name": name,
                "group": group,
                "status": "review",
                "observed": None,
                "limit": limit,
                "detail": "Metric is undefined because one or more groups lack the required class.",
            })
            return
        failed = value > limit if direction == "max" else value < limit
        findings.append({"name": name, "group": group, "status": "fail" if failed else "pass", "observed": value, "limit": limit})

    undefined_group = next((group_name for group_name, metrics in group_metrics.items() if metrics["true_positive_rate"] is None or metrics["false_positive_rate"] is None), None)
    record("demographic_parity_gap", fairness["demographic_parity_gap"], limits.max_demographic_parity_gap, "max")
    record("equal_opportunity_gap", eo_gap, limits.max_equal_opportunity_gap, "max", group=undefined_group)
    record("equalized_odds_gap", fairness["equalized_odds_gap"], limits.max_equalized_odds_gap, "max", group=undefined_group)
    record("selection_rate_ratio", fairness["selection_rate_ratio"], limits.min_selection_rate_ratio, "min")
    record("expected_calibration_error", float(overall["ece"]), limits.max_ece, "max")
    for group_name, metrics in group_metrics.items():
        if int(metrics["support"]) < limits.min_group_size:
            findings.append({
                "name": "minimum_group_support",
                "group": group_name,
                "status": "review",
                "observed": metrics["support"],
                "limit": limits.min_group_size,
            })
    passed = all(item["status"] == "pass" for item in findings)
    return AuditReport(len(labels), threshold, overall, group_metrics, fairness, findings, passed)