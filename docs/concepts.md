# Core concepts explained

This project is designed for checking binary classification models before deployment.

## 1. Binary classification

A binary classifier outputs one of two classes, usually represented as 0 or 1:

- spam vs not spam
- high risk vs low risk
- fraud vs not fraud
- default vs no default

The model often produces a probability between 0 and 1, and the decision threshold determines the final yes/no output.

For example, if the threshold is 0.5:

- score 0.81 -> predict positive class
- score 0.32 -> predict negative class

## 2. Probabilities vs predictions

A probability tells you confidence, not just the final decision.

A model may say:

- “I am 78% sure this example is positive”

That is different from:

- “I predict positive because the score is above the threshold”

This project checks both the final prediction and the underlying probability quality.

## 3. Accuracy and classification metrics

Accuracy is only one measure. It tells you how often the model is right overall, but it hides important differences between groups.

The audit includes:

- accuracy
- precision
- recall
- F1 score
- ROC AUC
- Brier score

These metrics help answer different questions:

- accuracy: overall correctness
- precision: how often positive predictions are correct
- recall: how many actual positives were found
- F1: combines precision and recall
- ROC AUC: how well the model separates classes
- Brier score: how accurate the predicted probabilities are

## 4. Calibration

Calibration asks whether the model’s confidence matches reality.

If the model says 0.8 for 100 examples, then roughly 80 of those examples should actually be positive.

If not, the model is miscalibrated.

The project computes Expected Calibration Error (ECE), which summarizes how far the model’s certainty deviates from observed outcomes.

## 5. Fairness

Fairness checks compare metrics across sensitive groups such as:

- gender
- age band
- region
- protected category

The project looks at:

- demographic parity gap
- equal opportunity gap
- equalized odds gap
- selection-rate ratio

It is important to understand that fairness is context-dependent. A gap may be acceptable in one setting and not in another. This package exposes the numbers so a team can review them with domain context.

## 6. Why minimum group size matters

When a group is tiny, the metric can be noisy or unstable.

The package allows a configurable `min_group_size` threshold. If a subgroup is too small, the finding is marked as review instead of treated like a confident pass/fail.

This prevents overreacting to random fluctuations in small populations.

## 7. Guardrails

A guardrail is a threshold for acceptable behavior.

Examples:

- maximum demographic parity gap of 0.10
- maximum ECE of 0.08
- minimum group size of 20

The audit reports whether the observed value passes, fails, or needs review.

This makes the package useful for deployment review and governance workflows.

## 8. Explainability and local attribution

Even a model with good aggregate metrics can behave unexpectedly for an individual input.

The local feature attribution logic checks which feature has the largest impact on one prediction by replacing one feature at a time with a reference value and measuring the resulting score difference.

This is a lightweight, model-agnostic way to answer:

- Why did the model give this score?
- Which input mattered most?
- Which feature pushed the prediction up or down?

## 9. Why this project is useful

The package helps explain and audit a model before deployment.

It is especially useful when:

- a model affects real people
- there are sensitive attributes in the data
- decisions must be documented and reviewed
- probability quality matters
- stakeholders want interpretable checks beyond raw accuracy

## 10. Important caution

These checks are descriptive and decision-support tools, not proof that a system is fair or safe.

Fairness metrics depend on:

- representative data
- well-defined groups
- context-specific thresholds
- legal and ethical review

A model should never be judged solely by an automated score.
