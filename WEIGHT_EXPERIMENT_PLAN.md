# KEIBALAB Weight Experiment Plan (No Production Changes)

Status: preparation only. This document does not change the live model, prediction scores, ticket selection, or web app.

## Goal

Determine whether the current model's ranking weakness is materially caused by feature-weight allocation. Compare challengers on identical races and identical available information, with the existing production configuration retained as the control.

## Existing scoring surfaces to audit

The historical scorer in `scripts/nar_backtest.py` exposes two distinct weight layers:

1. Core model weights: `form`, `venue`, `distance`, `sire`, `jockey`, `age`.
2. Recent-performance blend weights: `base`, `margin`, `closing`, `time`, `going` (the latter four are explicit inputs; base is the residual).

These layers must be tested separately first. Do not change both layers in the same experiment. The current live capture path obtains its model weights from the learned model artifact and calls the shared scoring functions; production remains untouched during this study.

## Experiment stages

### Stage 0 — Baseline and parity
- Freeze and record the exact main-branch commit, model artifact/version, weight vectors, temperature/calibration, feature availability rules, and race inclusion rules.
- Re-run the unmodified baseline and verify that it reproduces the currently published metrics.
- Confirm cached and uncached scoring parity for every tested weight vector.
- Reject any race with leakage, missing result linkage, invalid field mapping, or insufficient feature/odds coverage; report exclusions by reason.

### Stage 1 — One-factor-at-a-time sensitivity
- Core weights: change one of the six weights at a time while keeping the other five in their baseline proportions; normalize the vector to sum to 1.
- Performance blend: vary one of margin, closing, time, or going at a time, keeping the remaining blend weights fixed and base as the residual. Reject combinations whose explicit weights sum above 1.
- Use small, predeclared perturbations around baseline (for example relative multipliers 0.75, 0.90, 1.10, 1.25), not arbitrary hand-picked values.
- Report rank changes and feature contribution distributions, not only aggregate hit rate.

### Stage 2 — Limited interaction test
- Only after Stage 1 identifies plausible effects, test a small predeclared set of two-factor interactions.
- Keep the candidate count bounded. Do not run an unrestricted grid search over all combinations.
- Preserve identical race sets for every candidate so comparisons are paired.

### Stage 3 — Out-of-sample validation
- Use chronological splits only. Earlier dates are training/development; later dates are untouched validation/holdout.
- Fit/learn weights using training data only. Never use holdout results to choose weights, thresholds, or features.
- Keep a final prospective set from timestamped pre-off snapshots separate from historical final-odds backtests.
- Report sample counts and date ranges for every split. If a split has too few qualifying races, label it inconclusive rather than selecting a winner.

## Required metrics

Primary:
- Top-7 podium coverage: proportion of races where all three official trio finishers are inside the model's top seven.
- Mean number of official podium finishers captured in the top seven (0–3).
- Compare both metrics against the market ranking on exactly the same horses/races.

Secondary:
- Top-3 coverage / winner rank.
- Brier score and log loss where calibrated probabilities are available and valid.
- 3連複 12-ticket hit rate and ROI using the exact same fixed ticket-generation, odds timestamp, stake, and payout rules.
- Confidence intervals (race-level bootstrap) and paired per-race deltas versus baseline and market.

A candidate is not accepted merely because ROI or hit rate improves on a small sample. It must improve the primary ranking objective without a material deterioration in probability calibration or ticket performance, and the improvement must persist on untouched chronological data.

## Controls and safeguards

- Production weights, model artifact, prediction path, and ticket path are read-only throughout this experiment.
- Candidate weights are stored in experiment outputs with a unique experiment ID; never overwrite production model files.
- No final odds may be substituted for missing pre-off odds in prospective evaluation.
- Do not compare candidates on different race subsets without explicitly marking the comparison non-paired.
- Report missingness and feature availability by feature, venue, field size, and date.
- Check whether a feature is mostly neutral/missing; a weight cannot rescue a feature with poor data quality.
- Record every tested candidate, including unsuccessful candidates, to prevent repeated selection bias.

## Decision gate

No production promotion until:
1. baseline reproduction passes;
2. at least 100 qualifying untouched holdout races are available (more is preferable);
3. candidate beats baseline on top-7 podium coverage and mean podium capture on the same holdout races;
4. market comparison is reported, not omitted;
5. uncertainty intervals and all secondary metrics are reviewed;
6. a separate prospective shadow run confirms the result.

Until all gates pass, current production weights and ticket logic remain unchanged.
