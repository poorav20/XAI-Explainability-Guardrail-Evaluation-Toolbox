# Contributing

Thanks for helping improve this project.

## Development setup

```powershell
cd "d:\C\AI Agents\XAI project"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Run tests

```powershell
pytest
```

or:

```powershell
python -m pytest -q
```

## Run the demo

```powershell
python -m xai_guardrails.cli --output artifacts
```

## Pull request checklist

- Keep changes focused and minimal.
- Add or update tests when behavior changes.
- Verify the project still passes `pytest`.
- Document new behavior in the README or docs.
- Prefer model-agnostic logic and clear guardrail reasoning.

## Code philosophy

This project aims to be simple, explicit, and transparent:

- model-agnostic auditing
- no hidden framework assumptions
- clear fairness and calibration checks
- conservative review behavior when metrics are undefined
- reproducible reporting outputs

## Questions or ideas

Open a GitHub issue with a clear description, expected behavior, and a small reproducible example when possible.
