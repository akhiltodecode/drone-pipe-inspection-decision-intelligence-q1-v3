#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONHASHSEED=0
study_jobs="${DRONE_STUDY_JOBS:-4}"

python validate_pipeline.py
python -m unittest discover -v
python paper_pipeline.py \
  --clean \
  --replications 30 \
  --bootstrap-resamples 5000 \
  --jobs "$study_jobs" \
  --export-synthetic-data
python verify_outputs.py --output-dir outputs
