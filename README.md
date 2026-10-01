# XAI Guardrails

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![MIT License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-Pytest-orange)

A compact, model-agnostic toolkit for auditing binary classification models before deployment.
Build safer ML systems with transparent checks for quality, fairness, calibration, and explainability.

## Why teams use it

Machine learning models often look strong in aggregate while still being risky in practice. This project helps teams answer the questions that matter before deployment:

- Is the model accurate enough?
- Does it behave differently across groups?
- Are the probabilities trustworthy?
- Why did it make this decision?

## Project at a glance

- Model-agnostic API
- Binary classification audit tooling
- subgroup fairness checks
- calibration diagnostics
- local feature-attribution explanations
- CLI demo with JSON and plot outputs

## Key features

### Audit model quality

Evaluate the core binary-classification metrics:

- accuracy
- precision
- recall
- F1 score
- ROC AUC
- Brier score
- expected calibration error (ECE)

### Measure fairness

Compare group-wise performance for sensitive attributes such as race, gender, age band, region, or other protected categories:

- demographic parity gap
- equal opportunity gap
- equalized odds gap
- selection-rate ratio
- minimum subgroup support checks

### Monitor calibration

Test whether model confidence matches intended real-world outcomes, using ECE and related diagnostics.

### Explain individual predictions

See which feature most changed a score for a single prediction by replacing one feature at a time with a reference value.

## Quick demo

```powershell
cd "d:\C\AI Agents\XAI project"
python -m pip install -e ".[plots,test]"
python -m xai_guardrails.cli --output artifacts
```

This generates example outputs including:

- `artifacts/audit_report.json`
- `artifacts/group_metrics.png`

## Example output

The report includes overall metrics, per-group metrics, fairness findings, and review flags. It is designed to be easy to print, inspect, or save for governance workflows.

## Why this matters

When a model affects real people, a single aggregate metric is not enough. This project makes it easier to inspect the model from multiple angles before approving deployment.

## Getting started

### Installation

```powershell
cd "d:\C\AI Agents\XAI project"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[plots,test]"
```

### Run tests

```powershell
python -m pytest -q
```

### Use the library

```python
from xai_guardrails import AuditThresholds, audit_predictions

report = audit_predictions(
    y_true=y_test,
    y_score=model.predict_proba(X_test)[:, 1],
    sensitive_features=protected_group,
    thresholds=AuditThresholds(max_demographic_parity_gap=0.10, max_ece=0.08),
)

print(report.to_dict())
```

## Why this project exists

A model can look successful on average while still being risky in practice.

Common problems include:

- poor performance for a protected subgroup
- overconfident probability scores
- unfair differences in positive predictions across groups
- hard-to-explain individual decisions

This package turns those concerns into explicit audit checks and structured outputs.

## What it measures

The audit covers four major areas:

### 1. Overall model quality

- accuracy
- precision
- recall
- F1 score
- ROC AUC
- Brier score
- expected calibration error (ECE)

### 2. Subgroup fairness

- demographic parity gap
- equal opportunity gap
- equalized odds gap
- selection-rate ratio

### 3. Calibration quality

- how close predicted probabilities are to actual outcomes
- how large the expected calibration error is

### 4. Local explainability

- which feature most changes the score for a single prediction
- how much each feature pushes the probability up or down

## Core concepts explained simply

### Binary classification

The project is focused on binary tasks: yes/no or positive/negative predictions.

Examples:

- fraud vs no fraud
- default vs no default
- disease vs no disease
- spam vs not spam

### Probabilities vs predictions

A model usually outputs a probability between 0 and 1.

A threshold such as 0.5 converts that probability into a final decision.

Example:

- probability = 0.82 -> predicted positive
- probability = 0.31 -> predicted negative

The package checks both the probability values and the resulting decisions.

### Fairness

Fairness is measured by comparing metrics across groups.

If one group receives a much higher positive-prediction rate than another, the model may be unfair or reflect biased patterns in the data.

This project does not decide whether a disparity is acceptable; it makes the risk visible and structured.

### Calibration

Calibration checks whether a predicted probability is meaningful.

If the model says 0.8, then about 80% of those cases should actually be positive. If the real rate is far off, the model is poorly calibrated.

### Explainability

For a single prediction, the project can show which features had the most influence.

This helps answer questions like:

- Why was this person flagged?
- Which input factor most changed the decision?
- Was the model driven by a meaningful feature or a noisy one?

## Project status

This repository is a lightweight Python package intended for audit and review workflows, not as a replacement for domain review or legal process.

It is deliberately explicit about undefined or underpowered groups and raises them for review rather than silently treating them as passing.

## Installation

```powershell
cd "d:\C\AI Agents\XAI project"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[plots,test]"
```

The optional `plots` extra adds matplotlib for the demo plot. The `test` extra adds pytest.

## Quick start

### Run the demo CLI

```powershell
python -m xai_guardrails.cli --output artifacts
```

This creates synthetic example data, trains a small logistic regression model, and writes:

- `artifacts/audit_report.json`
- `artifacts/group_metrics.png`

### Use the package in Python

```python
from xai_guardrails import AuditThresholds, audit_predictions

report = audit_predictions(
    y_true=y_test,
    y_score=model.predict_proba(X_test)[:, 1],
    sensitive_features=protected_group,
    thresholds=AuditThresholds(max_demographic_parity_gap=0.10, max_ece=0.08),
)

print(report.to_dict())
```

This is the main API for auditing a real model.

## Local feature attribution example

```python
from xai_guardrails import local_feature_attributions

attributions = local_feature_attributions(
    model.predict_proba,
    X_test[0],
    reference=None,
)

for item in attributions[:5]:
    print(item.feature, item.score_change)
```

## Guardrail behavior

The audit uses configurable threshold values from `AuditThresholds`.

Example thresholds:

```python
from xai_guardrails import AuditThresholds

thresholds = AuditThresholds(
    max_demographic_parity_gap=0.10,
    max_equal_opportunity_gap=0.10,
    max_equalized_odds_gap=0.10,
    min_selection_rate_ratio=0.80,
    max_ece=0.10,
    min_group_size=20,
)
```

A finding can end up as:

- pass
- fail
- review

Review is used for edge cases such as small groups or undefined metrics.

## Project structure

```text
.
├── README.md
├── pyproject.toml
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
├── docs/
│   ├── concepts.md
│   └── architecture.md
├── src/
│   └── xai_guardrails/
│       ├── __init__.py
│       ├── audit.py
│       ├── cli.py
│       └── explain.py
└── tests/
    └── test_audit.py
```

## Running tests

```powershell
cd "d:\C\AI Agents\XAI project"
python -m pytest -q
```

The test suite covers:

- subgroup fairness detection
- calibration issues
- undefined group handling
- invalid input validation
- local attribution ranking

## Documentation

More detailed explanations are available in:

- [docs/concepts.md](docs/concepts.md)
- [docs/architecture.md](docs/architecture.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and pull request guidance.

## Community and safety

- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)
- [CHANGELOG.md](CHANGELOG.md)
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)

## License

This project is licensed under the [MIT License](LICENSE).

## Important note

This package is a decision-support and review tool. It helps expose risk, but it does not replace domain expertise, legal review, or responsible deployment judgment.

It is best used as part of a broader governance process for AI systems.
