#!/usr/bin/env python3
"""Independent post-run verification for scientific and publication artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_FIGURES = {"fig1_workflow.png", "fig2_empirical_results.png"}
EXPECTED_PAPER_TABLES = {
    "paper_table_1_study_design.csv",
    "paper_table_2_main_results.csv",
}
EXPECTED_METHODS = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(output_dir: Path) -> int:
    manifest = output_dir / "checksums.sha256"
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split("  ", 1)
        target = output_dir / filename
        if not target.is_file():
            raise AssertionError(f"manifest target is missing: {filename}")
        observed = sha256(target)
        if observed != expected:
            raise AssertionError(f"checksum mismatch: {filename}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    stale = list(output_dir.glob(".*.tmp"))
    stale.extend((output_dir / "checkpoints").glob(".*.tmp"))
    assert not stale, f"stale checkpoint temporary files: {[path.name for path in stale]}"

    figures = {path.name for path in output_dir.glob("fig*.png")}
    paper_tables = {path.name for path in output_dir.glob("paper_table_*.csv")}
    assert figures == EXPECTED_FIGURES, figures
    assert paper_tables == EXPECTED_PAPER_TABLES, paper_tables

    run_state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))
    assert run_state["status"] == "completed"
    assert run_state["replications_completed"] == run_state["replications_requested"]

    environment = json.loads((output_dir / "environment.json").read_text(encoding="utf-8"))
    replications = int(environment["replications_executed"])
    assert replications == run_state["replications_completed"]

    final = pd.read_csv(output_dir / "final_results.csv")
    main_results = final[final["scenario"].eq("main")]
    assert main_results["method"].nunique() == EXPECTED_METHODS
    assert main_results["replications"].eq(replications).all()
    if replications >= 2:
        assert main_results[["ci95_low", "ci95_high"]].notna().all().all()
        assert (main_results["ci95_low"] <= main_results["mean"]).all()
        assert (main_results["mean"] <= main_results["ci95_high"]).all()

    predictions = pd.read_csv(output_dir / "test_predictions.csv")
    assert predictions["method"].nunique() == EXPECTED_METHODS
    assert predictions["predicted_probability"].between(0, 1).all()
    assert predictions["replicate"].nunique() == replications

    sensitivity = pd.read_csv(output_dir / "sensitivity_summary.csv")
    assert not sensitivity.empty
    assert sensitivity["scenario"].str.startswith(("sensor_stress", "cost_")).all()

    checked = verify_checksums(output_dir)
    print(f"PASS: {replications} completed replications")
    print(f"PASS: exactly {len(figures)} paper figures and {len(paper_tables)} paper tables")
    print(f"PASS: six methods with bounded held-out probabilities")
    print(f"PASS: {checked} scientific checksums verified")


if __name__ == "__main__":
    main()
