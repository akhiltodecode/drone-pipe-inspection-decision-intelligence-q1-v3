# Q1-Level Code and Experimental Audit

## Scope and interpretation

This audit covers the complete executable research path: synthetic data generation, grouped splitting, preprocessing, model fitting, calibration, risk-to-action optimization, stress tests, statistical aggregation, figures, tables, and deterministic output verification. It is a rigorous software and internal-validity audit, not a claim of mathematical formal verification or external field validation.

## Baseline checks completed

- Python syntax compilation passed.
- Every locked dependency exactly matched the execution environment.
- No API keys, passwords, tokens, network requests, subprocess execution, dynamic evaluation, or unsafe deserialization were present in the research code.
- Repeated generation with an identical seed produced identical data.
- Asset identifiers did not cross training, calibration, and test partitions.
- Prohibited simulator variables did not enter model features.
- All six policies returned finite probabilities in `[0, 1]` and valid actions.
- Frozen scientific outputs passed their SHA-256 manifest.

## Material weakness found and corrected

The original paired bootstrap resampled assets within each Monte Carlo replication but did not resample the replications themselves. That procedure could understate uncertainty because it treated the realized replication set as fixed. Version 2 uses a hierarchical paired bootstrap that resamples both replications and assets. It also reports replication-level Wilcoxon tests with Holm multiplicity adjustment.

## Q1-oriented improvements implemented

1. Expanded the prespecified full study from 10 to 30 independent Monte Carlo replications while retaining the original first ten seeds.
2. Added replication-level standard errors and 95% t confidence intervals for every reported metric.
3. Added hierarchical paired bootstrap confidence intervals, probability of improvement, Wilcoxon p-values, and Holm-adjusted p-values.
4. Added optimizer invariance tests over 1,001 risk values for every criticality class.
5. Added explicit tests for fixed-rule boundaries and sensor-stress directionality.
6. Added exactly two paper figures: the workflow and a two-panel empirical performance/calibration figure.
7. Added exactly two paper tables: study design and main held-out results.
8. Added a one-command full-study script, continuous integration validation, citation metadata, and an Apache 2.0 license.

## Version 3 computational hardening

1. Centralized every locked study, seed, action-cost, stress, and publication-limit assumption in `study_config.json`, with fail-fast schema and consistency checks.
2. Split the stable CLI, configuration, runtime safety, experiment pipeline, tests, and output verifier into explicit modules.
3. Added atomic final-artifact writes and portable compressed-JSON checkpoints with replication/seed metadata validation.
4. Added `--resume` and deterministic process-level `--jobs` execution. Numerical-library threads are limited to one per worker to prevent BLAS/OpenMP oversubscription.
5. Added nine automated unit tests plus the original end-to-end scientific contract validator.
6. Added independent post-run validation of output schemas, probabilities, replication counts, checksums, and the exact two-figure/two-table paper limit.
7. Added supplementary `sensitivity_summary.csv` while leaving the paper-facing output at two figures and two tables.
8. Expanded continuous integration to include parallel smoke execution, output verification, linting, type checks, and coverage-capable tests.

## Claim boundary

The experiment provides reproducible internal evidence for a simulation framework. It does not establish field defect-detection accuracy, real maintenance savings, transfer to a specific industrial plant, causal intervention effects, regulatory compliance, or autonomous shutdown authority.

## Full-study result verification

- 30 replications produced 108,000 unique segment inspections from 9,000 simulated assets.
- The exported data contained no duplicate `(replicate, asset_id, segment_id)` keys and no non-finite observed numeric values.
- The simulated failure prevalence was 15.37%; thermal and RGB-crack missingness were 12.72% and 5.64%, respectively.
- Calibrated random forest achieved the lowest nominal MEL: 19.8928 (95% CI 18.8574–20.9283).
- Calibrated logistic regression was close: 19.9375 (95% CI 18.9109–20.9641).
- The calibrated GB optimizer improved on uncalibrated GB by 1.7175 MEL units and on calibrated GB with fixed thresholds by 0.4055 units in the hierarchical paired analysis.
- The calibrated GB optimizer did not outperform the fixed heuristic, calibrated logistic regression, or calibrated random forest. This negative result is retained and must be stated plainly.
- Sensor stress increased MEL for every policy, as expected, without causing invalid probabilities or actions.
- A second independent 30-replication execution reproduced all 14 scientific artifacts byte-for-byte; no scientific file was missing, extra, or changed.

## Data needed for external validation

No external dataset is required to complete the simulation study. A stronger field-validity extension would require de-identified, asset-grouped inspection records containing RGB/thermal observations, acquisition-quality metadata, pipe context, inspection timestamps, and independently adjudicated defect or failure outcomes. Until such data are available, the manuscript must identify every dataset, cost, and outcome as synthetic.
