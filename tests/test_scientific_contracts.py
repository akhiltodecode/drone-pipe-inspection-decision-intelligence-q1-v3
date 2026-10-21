"""Fast, dependency-light scientific contract tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from drone_study.config import CONFIG, CONFIG_SHA256
from drone_study.pipeline import (
    ACTIONS,
    FEATURE_COLS,
    PROHIBITED,
    fixed_actions,
    generate_replication,
    grouped_split,
    optimal_actions,
    public_dataset,
    realized_expected_loss,
    stress_features,
)


class ScientificContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = generate_replication(seed=101, replicate=1)

    def test_configuration_is_locked_and_complete(self) -> None:
        self.assertEqual(len(CONFIG_SHA256), 64)
        self.assertEqual(CONFIG["publication_limits"]["maximum_paper_figures"], 2)
        self.assertEqual(CONFIG["publication_limits"]["maximum_paper_tables"], 2)
        self.assertFalse(set(FEATURE_COLS) & PROHIBITED)

    def test_simulator_is_deterministic(self) -> None:
        repeated = generate_replication(seed=101, replicate=1)
        pd.testing.assert_frame_equal(self.data, repeated, check_exact=True)

    def test_grouped_split_has_no_asset_overlap(self) -> None:
        train, calibration, test = grouped_split(self.data, seed=101)
        groups = [set(frame["asset_id"]) for frame in (train, calibration, test)]
        self.assertTrue(groups[0].isdisjoint(groups[1]))
        self.assertTrue(groups[0].isdisjoint(groups[2]))
        self.assertTrue(groups[1].isdisjoint(groups[2]))

    def test_public_export_removes_simulator_only_fields(self) -> None:
        exported = public_dataset(self.data)
        hidden = {
            "true_probability", "latent_corrosion", "latent_crack",
            "latent_leak", "asset_effect",
        }
        self.assertTrue(hidden.isdisjoint(exported.columns))

    def test_threshold_boundary_contract(self) -> None:
        probability = np.array([0.0, 0.0999, 0.10, 0.2499, 0.25, 0.5499, 0.55, 1.0])
        self.assertEqual(fixed_actions(probability).tolist(), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_optimizer_is_monotone_and_loss_minimizing(self) -> None:
        risk = np.linspace(0, 1, 1001)
        for criticality in ("low", "medium", "high"):
            levels = pd.Series([criticality] * len(risk))
            chosen_actions = optimal_actions(risk, levels)
            self.assertTrue((np.diff(chosen_actions) >= 0).all())
            self.assertTrue(set(np.unique(chosen_actions)).issubset(range(len(ACTIONS))))
            chosen_loss = realized_expected_loss(risk, chosen_actions, levels)
            for candidate in range(len(ACTIONS)):
                candidate_actions = np.full(len(risk), candidate, dtype=int)
                candidate_loss = realized_expected_loss(risk, candidate_actions, levels)
                self.assertTrue((chosen_loss <= candidate_loss + 1e-12).all())

    def test_stress_scenario_degrades_acquisition_quality(self) -> None:
        _, _, test = grouped_split(self.data, seed=101)
        stressed = stress_features(test, seed=101)
        thermal = ["thermal_delta", "thermal_hotspot", "thermal_gradient"]
        target = CONFIG["robustness"]["stress_thermal_missing_fraction"]
        self.assertGreaterEqual(stressed[thermal].isna().all(axis=1).mean(), target)
        self.assertGreaterEqual(stressed["blur"].mean(), test["blur"].mean())
        self.assertLessEqual(
            stressed["thermal_confidence"].mean(), test["thermal_confidence"].mean()
        )


if __name__ == "__main__":
    unittest.main()
