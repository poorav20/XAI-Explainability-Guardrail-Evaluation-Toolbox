# Project architecture

This repository is intentionally compact and modular.

## Package layout

```text
xai_guardrails/
├── __init__.py
├── audit.py
├── cli.py
├── explain.py

tests/
├── test_audit.py

README.md
pyproject.toml
CONTRIBUTING.md
LICENSE
```

## Main modules

### `audit.py`

This file contains the core model auditing logic.

It provides:

- `AuditThresholds`: configurable guardrail limits
- `AuditReport`: structured results container
- `audit_predictions(...)`: main evaluation entry point

The audit flow is:

1. validate inputs
2. compute score-level metrics
3. compute within-group metrics
4. compute fairness gaps
5. check thresholds
6. produce findings and pass/fail status

### `explain.py`

This file provides `local_feature_attributions(...)`.

It works by:

- taking one row of data
- measuring the base model score for that row
- replacing each feature with a reference value
- recomputing the score
- recording the score change

The output is a ranked list of feature importance values for that prediction.

### `cli.py`

This file runs an end-to-end demo that:

- creates synthetic classification data
- trains a logistic regression model
- evaluates the model with the audit logic
- writes a JSON report
- saves a plot of selection rates by group

This is useful for demonstration and for verifying the package works as intended.

## Data flow

```text
input labels + scores + group labels
        ↓
validate inputs
        ↓
compute overall metrics
        ↓
compute group metrics
        ↓
compare groups for fairness gaps
        ↓
check thresholds and guardrails
        ↓
return AuditReport
```

## Why this architecture works

The repository keeps concerns separated:

- auditing logic lives in one place
- explanation logic lives in another
- CLI logic is isolated for demo usage
- tests validate expected behavior without depending on the UI or packaging layer

This makes the package easier to understand, easier to extend, and easier to test.

## Design goals

- model-agnostic input handling
- deterministic, transparent metrics
- explicit review rather than silent pass/fail
- minimal dependencies
- easy CLI demo and Python API usage

## Extension points

If you want to extend the project, common additions include:

- additional fairness metrics
- calibration plots
- subgroup visualizations
- richer explanation methods
- report export formats such as CSV or HTML

The current structure makes those additions straightforward without breaking the core API.
