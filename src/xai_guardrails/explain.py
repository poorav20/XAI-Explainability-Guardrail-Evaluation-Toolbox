"""Model-agnostic local feature attribution by feature replacement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class FeatureAttribution:
    feature: str
    reference_value: Any
    observed_value: Any
    score_change: float


def _frame_values(row: Any) -> tuple[np.ndarray, list[str], bool]:
    if hasattr(row, "columns") and hasattr(row, "iloc"):
        if len(row) != 1:
            raise ValueError("row must contain exactly one sample")
        return row.iloc[0].to_numpy(copy=True), [str(name) for name in row.columns], True
    values = np.asarray(row)
    if values.ndim == 1:
        values = values.copy()
    elif values.ndim == 2 and values.shape[0] == 1:
        values = values[0].copy()
    else:
        raise ValueError("row must be a 1D array or a single-row frame")
    return values, [f"feature_{index}" for index in range(values.size)], False


def local_feature_attributions(
    predict_scores: Callable[[Any], Any],
    row: Any,
    *,
    reference: Any | None = None,
) -> list[FeatureAttribution]:
    """Measure score changes when each feature is replaced by a reference value.

    The callback must return one positive-class probability per input sample. The
    score change is observed prediction minus prediction with that feature replaced.
    """
    values, names, is_frame = _frame_values(row)
    if reference is None:
        if is_frame:
            reference_values = row.median(numeric_only=True).reindex(row.columns).to_numpy()
            for index, value in enumerate(reference_values):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    reference_values[index] = row.iloc[0, index]
        else:
            try:
                reference_values = np.nanmedian(values.astype(float), axis=0)
            except (TypeError, ValueError):
                reference_values = values.copy()
    else:
        reference_values, _, _ = _frame_values(reference) if hasattr(reference, "columns") else (np.asarray(reference).reshape(-1), names, False)
        if len(reference_values) != len(values):
            raise ValueError("reference must have the same number of features as row")

    def sample_for(candidate: np.ndarray) -> Any:
        if is_frame:
            result = row.copy()
            result.iloc[0, :] = candidate
            return result
        return candidate.reshape(1, -1)

    base_score = float(np.asarray(predict_scores(sample_for(values))).reshape(-1)[0])
    attributions: list[FeatureAttribution] = []
    for index, name in enumerate(names):
        changed = values.copy()
        changed[index] = reference_values[index]
        replaced_score = float(np.asarray(predict_scores(sample_for(changed))).reshape(-1)[0])
        attributions.append(FeatureAttribution(name, reference_values[index].item() if hasattr(reference_values[index], "item") else reference_values[index], values[index].item() if hasattr(values[index], "item") else values[index], base_score - replaced_score))
    return sorted(attributions, key=lambda item: abs(item.score_change), reverse=True)