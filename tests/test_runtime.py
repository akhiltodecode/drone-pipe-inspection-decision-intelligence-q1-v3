"""Crash-safety and checkpoint validation tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from drone_study.runtime import (
    atomic_write_csv,
    atomic_write_json,
    cleanup_stale_temporary_files,
    read_checkpoint,
    write_checkpoint,
)


class RuntimeTests(unittest.TestCase):
    def test_atomic_writers_produce_complete_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(root / "value.json", {"value": 3})
            self.assertEqual(json.loads((root / "value.json").read_text()), {"value": 3})
            frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
            atomic_write_csv(frame, root / "frame.csv")
            pd.testing.assert_frame_equal(pd.read_csv(root / "frame.csv"), frame)

    def test_checkpoint_round_trip_and_metadata_guard(self) -> None:
        bundle = {
            "replicate": 1,
            "data_seed": 101,
            "model_seed": 17,
            "raw_rows": [{"value": 1.5}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replication_01.json.gz"
            write_checkpoint(path, bundle)
            self.assertEqual(read_checkpoint(path, 1, 101, 17), bundle)
            with self.assertRaises(ValueError):
                read_checkpoint(path, 2, 101, 17)

    def test_stale_temporary_cleanup_is_narrowly_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = root / ".replication_01.json.gz.123.tmp"
            unrelated = root / "keep.tmp"
            stale.write_text("partial", encoding="utf-8")
            unrelated.write_text("keep", encoding="utf-8")
            self.assertEqual(cleanup_stale_temporary_files(root), 1)
            self.assertFalse(stale.exists())
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
