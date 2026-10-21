#!/usr/bin/env python3
"""Fast contract validation for the drone-inspection research pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from drone_study.config import CONFIG, CONFIG_SHA256
from drone_study.pipeline import (
    ACTIONS,
    FEATURE_COLS,
    PROHIBITED,
    fit_models,
    fixed_actions,
    generate_replication,
    grouped_split,
    optimal_actions,
    predict_all,
    public_dataset,
    realized_expected_loss,
    stress_features,
    summarize_metrics,
)


def main() -> None:
    data = generate_replication(seed=101, replicate=1)
    repeated = generate_replication(seed=101, replicate=1)
    assert data.equals(repeated), "simulation is not deterministic"
    study = CONFIG["study"]
    assert len(CONFIG_SHA256) == 64
    assert len(data) == study["assets_per_replication"] * study["segments_per_asset"]
    assert data["asset_id"].nunique() == study["assets_per_replication"]
    assert (data.groupby("asset_id").size() == study["segments_per_asset"]).all()

    train, calibration, test = grouped_split(data, seed=101)
    asset_sets = [set(frame["asset_id"]) for frame in (train, calibration, test)]
    assert not (asset_sets[0] & asset_sets[1])
    assert not (asset_sets[0] & asset_sets[2])
    assert not (asset_sets[1] & asset_sets[2])
    expected_rows = [
        study["train_assets"] * study["segments_per_asset"],
        study["calibration_assets"] * study["segments_per_asset"],
        study["test_assets"] * study["segments_per_asset"],
    ]
    assert [len(frame) for frame in (train, calibration, test)] == expected_rows
    assert not (set(FEATURE_COLS) & PROHIBITED), "model feature leakage detected"

    exported = public_dataset(data)
    assert len(exported) == len(data)
    assert not ({"true_probability", "latent_corrosion", "latent_crack",
                 "latent_leak", "asset_effect"} & set(exported.columns))

    with threadpool_limits(limits=1):
        lr, rf, gb, calibrators = fit_models(train, calibration, seed=17)
        predictions, _ = predict_all(test, train, lr, rf, gb, calibrators)
    assert len(predictions) == 6
    for name, probability in predictions.items():
        assert probability.shape == (len(test),), name
        assert np.isfinite(probability).all(), name
        assert ((probability >= 0) & (probability <= 1)).all(), name
        action = optimal_actions(probability, test["criticality"])
        assert action.shape == (len(test),), name
        assert set(np.unique(action)).issubset(range(len(ACTIONS))), name

    # Threshold boundary contract for the author-defined rule.
    boundary_actions = fixed_actions(np.array([0.0, 0.0999, 0.10, 0.2499, 0.25, 0.5499, 0.55, 1.0]))
    assert boundary_actions.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]

    # The optimizer must choose an action that minimizes the implemented loss,
    # and intervention intensity must not decrease as risk increases.
    risk_grid = np.linspace(0, 1, 1001)
    for criticality in ["low", "medium", "high"]:
        criticality_grid = pd.Series([criticality] * len(risk_grid))
        actions = optimal_actions(risk_grid, criticality_grid)
        assert (np.diff(actions) >= 0).all(), criticality
        chosen = realized_expected_loss(risk_grid, actions, criticality_grid)
        for candidate in range(len(ACTIONS)):
            candidate_actions = np.full(len(risk_grid), candidate, dtype=int)
            candidate_loss = realized_expected_loss(risk_grid, candidate_actions, criticality_grid)
            assert (chosen <= candidate_loss + 1e-12).all(), criticality

    stressed = stress_features(test, seed=101)
    thermal_cols = ["thermal_delta", "thermal_hotspot", "thermal_gradient"]
    assert stressed[thermal_cols].isna().all(axis=1).mean() >= 0.25
    assert stressed["blur"].mean() >= test["blur"].mean()
    assert stressed["thermal_confidence"].mean() <= test["thermal_confidence"].mean()

    # Replication-level summary intervals must contain their point estimate.
    toy = pd.DataFrame({
        "scenario": ["main"] * 4,
        "method": ["m"] * 4,
        "metric": ["MEL"] * 4,
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    summary = summarize_metrics(toy).iloc[0]
    assert summary.ci95_low < summary["mean"] < summary.ci95_high

    print("PASS: deterministic simulation")
    print(f"PASS: grouped asset split ({'/'.join(map(str, expected_rows))} rows)")
    print("PASS: no prohibited model features")
    print("PASS: leakage-safe public dataset schema")
    print("PASS: six prediction policies and valid decision actions")
    print("PASS: threshold boundaries and expected-loss optimizer invariants")
    print("PASS: sensor-stress direction and missingness contract")
    print("PASS: replication-level confidence interval contract")


if __name__ == "__main__":
    main()
