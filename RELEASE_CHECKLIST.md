# Release checklist

Use this checklist before publishing a new release.

## Pre-release

- [ ] Confirm the package version in `pyproject.toml` is correct
- [ ] Review the changelog and summarize the release highlights
- [ ] Run the full test suite: `python -m pytest -q`
- [ ] Confirm install works in a clean environment
- [ ] Verify CLI demo output runs successfully
- [ ] Check README examples still work as written
- [ ] Ensure optional dependencies still install correctly

## Documentation

- [ ] README reflects the current project state
- [ ] Feature list matches actual behavior
- [ ] Examples are valid and reproducible
- [ ] Security and conduct docs are in place

## Packaging

- [ ] Build the package locally
- [ ] Confirm metadata is correct
- [ ] Check for unexpected files in the release bundle

## Publish

- [ ] Tag the release commit
- [ ] Publish to package index if applicable
- [ ] Add release notes to GitHub
- [ ] Verify the project page renders correctly
