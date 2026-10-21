# RunPod Execution Guide

## Resource choice

The current experiment uses NumPy, pandas, SciPy, and scikit-learn. It does not use CUDA, PyTorch, TensorFlow, or another GPU execution path. A CPU instance with at least 4 vCPUs, 16 GB RAM, and 25 GB storage is sufficient. A GPU pod will run the code but will not accelerate the present models.

## Clean execution

```bash
git clone https://github.com/akhiltodecode/drone-pipe-inspection-ai-decision-intelligence.git
cd drone-pipe-inspection-ai-decision-intelligence

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

python validate_pipeline.py
python -m unittest discover -v
DRONE_STUDY_JOBS=4 bash run_full_study.sh
```

## Completion criteria

The run is complete only when:

1. all eight validation lines begin with `PASS`;
2. the full study reports 30 replications;
3. `verify_outputs.py` verifies every scientific checksum;
4. `outputs/environment.json` records the expected package versions and `PYTHONHASHSEED=0`;
5. `outputs/fig1_workflow.png`, `outputs/fig2_empirical_results.png`, and both paper tables exist.

`outputs/timing_results.csv` is hardware-dependent and intentionally excluded from the scientific checksum manifest.

## Resume after interruption

Do not use `--clean` when resuming. Run:

```bash
PYTHONHASHSEED=0 python paper_pipeline.py \
  --resume \
  --replications 30 \
  --bootstrap-resamples 5000 \
  --jobs 4 \
  --export-synthetic-data
python verify_outputs.py --output-dir outputs
```

Use no more workers than the CPU cores and available memory support. Four workers is the recommended starting point for a 4-vCPU/16-GB CPU pod. The project does not use CUDA, so purchasing a larger GPU does not improve this study.
