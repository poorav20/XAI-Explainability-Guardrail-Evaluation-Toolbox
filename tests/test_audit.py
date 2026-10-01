import numpy as np
import pytest

from xai_guardrails import AuditThresholds, audit_predictions, local_feature_attributions


def test_detects_subgroup_fairness_and_calibration_issues():
    labels = np.array([1, 1, 0, 0] * 30)
    groups = np.array(["a"] * 60 + ["b"] * 60)
    scores = np.concatenate([np.full(60, 0.9), np.full(60, 0.1)])
    report = audit_predictions(
        labels,
        scores,
        groups,
        thresholds=AuditThresholds(min_group_size=10, max_ece=0.05),
    )
    assert not report.passed
    assert report.fairness["demographic_parity_gap"] == 1.0
    assert report.fairness["equal_opportunity_gap"] == 1.0
    assert report.overall["ece"] > 0.05
    assert any(item["status"] == "fail" for item in report.findings)


def test_flags_undefined_group_rate_and_small_support_for_review():
    report = audit_predictions(
        [1, 1, 0, 0, 1],
        [0.8, 0.7, 0.2, 0.3, 0.6],
        ["a", "a", "a", "a", "b"],
        thresholds=AuditThresholds(min_group_size=2),
    )
    assert any(item["status"] == "review" and item["group"] == "b" for item in report.findings)
    assert report.groups["b"]["false_positive_rate"] is None
    assert any(item["name"] == "equal_opportunity_gap" and item["status"] == "review" for item in report.findings)


def test_rejects_non_probability_scores_and_mismatched_inputs():
    with pytest.raises(ValueError, match="probabilities"):
        audit_predictions([0, 1], [-0.1, 1.1], ["a", "b"])
    with pytest.raises(ValueError, match="equal lengths"):
        audit_predictions([0, 1], [0.2, 0.8], ["a"])


def test_local_attribution_ranks_a_feature_that_changes_the_score():
    def predict_scores(rows):
        values = np.asarray(rows)
        return 1 / (1 + np.exp(-values[:, 0]))

    attributions = local_feature_attributions(predict_scores, np.array([2.0, 0.5]), reference=[0.0, 0.0])
    assert attributions[0].feature == "feature_0"
    assert attributions[0].score_change > 0
    assert attributions[1].score_change == 0