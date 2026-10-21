"""Crash-safe experiment I/O, logging, checkpoints, and provenance helpers."""

from __future__ import annotations

import gzip
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CHECKPOINT_SCHEMA_VERSION = 1


def cleanup_stale_temporary_files(directory: Path) -> int:
    """Remove only this pipeline's hidden atomic-write remnants."""
    removed = 0
    if directory.is_dir():
        for path in directory.glob(".*.tmp"):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure one concise process-safe console logger."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
    return logging.getLogger("drone_study")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a text artifact atomically after a successful complete write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_checkpoint(path: Path, bundle: dict[str, Any]) -> None:
    """Write a portable, non-pickle gzip JSON checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "bundle": bundle}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(_json_safe(payload), handle, separators=(",", ":"), allow_nan=False)
    os.replace(temporary, path)


def read_checkpoint(path: Path, replicate: int, data_seed: int, model_seed: int) -> dict[str, Any]:
    """Read and authenticate checkpoint metadata before accepting its records."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {path}")
    bundle = payload["bundle"]
    expected = (replicate, data_seed, model_seed)
    observed = (bundle.get("replicate"), bundle.get("data_seed"), bundle.get("model_seed"))
    if observed != expected:
        raise ValueError(f"checkpoint metadata mismatch for {path}: {observed} != {expected}")
    return bundle


def git_commit(project_root: Path) -> str:
    """Resolve a local Git commit without shelling out or accessing a network."""
    git_dir = project_root / ".git"
    head = git_dir / "HEAD"
    if not head.is_file():
        return "not-a-git-checkout"
    value = head.read_text(encoding="utf-8").strip()
    if not value.startswith("ref: "):
        return value
    ref = value.removeprefix("ref: ")
    loose_ref = git_dir / ref
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == ref:
                    return commit
    return f"unresolved:{ref}"


def runtime_metadata(started_at: float, project_root: Path) -> dict[str, Any]:
    """Return operational provenance; timing stays outside scientific checksums."""
    return {
        "elapsed_seconds": time.perf_counter() - started_at,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "git_commit": git_commit(project_root),
        "command": sys.argv,
    }
