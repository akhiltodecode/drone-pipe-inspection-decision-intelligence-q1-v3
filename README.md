# Code Package: Drone Inspection Data, AI/ML, and Decision Intelligence

## Working research title

**Data-Driven AI and Machine Learning for Drone-Based Industrial Pipe Inspection: A Simulation Framework for Maintenance Decision Intelligence**

This package implements the computational study only. It does not generate or revise the final paper. Version 3 adds crash-safe resumability, deterministic multi-process execution, a centralized protocol configuration, automated tests, provenance, and artifact verification while preserving the simulation-only evidence boundary.

## What the code does

1. Simulates 30 reproducible datasets containing 300 pipe assets and 12 inspected segments per asset (108,000 segment inspections total).
2. Creates RGB, thermal, pipe-context, flight-quality, and 12-month failure variables with realistic noise and missingness.
3. Separates each replication by pipe asset into training, calibration, and held-out test sets (70%/15%/15%).
4. Fits logistic regression, random forest, and histogram gradient boosting models, with sigmoid probability calibration where specified.
5. Converts predicted risk into monitor, reinspect, planned repair, or urgent shutdown actions by minimizing simulated expected loss.
6. Evaluates probability quality, ranking, maintenance loss, regret, stress conditions, cost sensitivity, and fixed-rule baselines.
7. Quantifies uncertainty with replication-level confidence intervals and a hierarchical paired bootstrap over replications and assets.

## Files

- `paper_pipeline.py`: stable command-line entry point.
- `drone_study/pipeline.py`: simulation, modeling, decision, statistical, and reporting pipeline.
- `drone_study/config.py` and `study_config.json`: validated, versioned scientific assumptions.
- `drone_study/runtime.py`: atomic writes, safe compressed-JSON checkpoints, logging, and provenance.
- `tests/`: automated scientific-contract and crash-safety tests.
- `validate_pipeline.py`: one-replication contract and leakage smoke test.
- `verify_outputs.py`: independent checksum, schema, and two-figure/two-table verification.
- `run_full_study.sh`: one-command 30-replication execution and checksum verification.
- `Q1_CODE_AUDIT.md`: audit findings, corrections, and claim boundaries.
- `RUNPOD.md`: CPU-oriented RunPod execution and completion criteria.
- `Dockerfile`: containerized Python 3.12 execution path.
- `requirements-lock.txt`: exact Python dependency versions.
- `outputs/synthetic_inspections.csv`: analysis-ready simulated dataset; latent generator variables are excluded.
- `outputs/final_results.csv`: mean, standard deviation, standard error, and 95% confidence interval across replications.
- `outputs/raw_results.csv`: replication-level metrics.
- `outputs/paired_comparisons.csv`: hierarchical paired-bootstrap comparisons and multiplicity-adjusted tests.
- `outputs/test_predictions.csv`: held-out probabilities, actions, and losses.
- `outputs/split_audit.csv` and `outputs/data_audit.csv`: split and data-quality evidence.
- `outputs/checksums.sha256`: deterministic output hashes; timing is intentionally excluded.
- `outputs/fig1_workflow.png` and `outputs/fig2_empirical_results.png`: the only two paper figures.
- `outputs/paper_table_1_study_design.csv` and `outputs/paper_table_2_main_results.csv`: the only two paper tables.

## Run

Create an isolated Python environment, install the locked requirements, and execute:

```bash
python -m pip install -r requirements-lock.txt
python validate_pipeline.py
python -m unittest discover -v
PYTHONHASHSEED=0 python paper_pipeline.py --clean --replications 30 --bootstrap-resamples 5000 --jobs 4 --export-synthetic-data
python verify_outputs.py --output-dir outputs
```

On Linux or RunPod, the equivalent one-command study is:

```bash
bash run_full_study.sh
```

The `--clean` option is deliberately restricted to this package's `outputs` directory. Use `--replications 2 --bootstrap-resamples 1000 --jobs 2` for a fast smoke run; only the 30-replication command is intended for final reporting. `--jobs` controls independent replication workers, and native numerical threads are limited to one per worker to prevent CPU oversubscription.

If a run is interrupted, use the same scientific arguments with `--resume` instead of `--clean`:

```bash
PYTHONHASHSEED=0 python paper_pipeline.py --resume --replications 30 --bootstrap-resamples 5000 --jobs 4 --export-synthetic-data
```

Completed replication checkpoints are validated against replication and seed metadata before reuse. Checkpoints use compressed JSON rather than executable pickle serialization.

## Leakage controls

Models receive only the variables in `FEATURE_COLS`. They do not receive asset or segment identifiers, the outcome, the data-generating probability, latent corrosion/crack/leak states, or the asset random effect. Pipe assets cannot cross training, calibration, and test partitions.

## Interpretation boundary

All observations, failures, costs, and maintenance effects are simulated. The code supports reproducible methodological evidence, not claims of field deployment, real operational savings, or autonomous shutdown authority.
