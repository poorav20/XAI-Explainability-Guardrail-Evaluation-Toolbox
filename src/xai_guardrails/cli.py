"""Run an end-to-end synthetic model audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .audit import AuditThresholds, audit_predictions
from .explain import local_feature_attributions


def _plot_groups(report: Any, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install the optional plotting dependencies with: pip install -e '.[plots]'") from exc
    group_names = list(report.groups)
    rates = [report.groups[name]["selection_rate"] for name in group_names]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(group_names, rates, color="#287a70")
    axis.set(title="Positive prediction rate by group", ylabel="Selection rate", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_demo(output_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(24)
    features, labels = make_classification(
        n_samples=1400,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.63, 0.37],
        class_sep=0.85,
        random_state=24,
    )
    sensitive = rng.choice(["group_a", "group_b"], size=len(labels), p=[0.65, 0.35])
    features[:, 0] += (sensitive == "group_b") * 0.85
    labels = ((features[:, 0] + features[:, 1] * 0.25) > np.quantile(features[:, 0] + features[:, 1] * 0.25, 0.58)).astype(int)
    train_x, test_x, train_y, test_y, _, test_groups = train_test_split(
        features, labels, sensitive, test_size=0.3, random_state=24, stratify=labels
    )
    model = LogisticRegression(max_iter=500).fit(train_x, train_y)
    scores = model.predict_proba(test_x)[:, 1]
    report = audit_predictions(
        test_y,
        scores,
        test_groups,
        thresholds=AuditThresholds(min_group_size=30),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_data = report.to_dict()
    report_data["example_attributions"] = [
        attribution.__dict__
        for attribution in local_feature_attributions(model.predict_proba, test_x[0])
    ]
    report_path = output_dir / "audit_report.json"
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    _plot_groups(report, output_dir / "group_metrics.png")
    return {"report": report, "report_path": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the XAI guardrail evaluation demo")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Directory for JSON and plot outputs")
    args = parser.parse_args()
    result = run_demo(args.output)
    report = result["report"]
    print(f"Audit {'PASS' if report.passed else 'REVIEW'}: {result['report_path']}")
    for finding in report.findings:
        print(f"{finding['status'].upper():6} {finding['name']}: {finding['observed']} (limit {finding['limit']})")


if __name__ == "__main__":
    main()