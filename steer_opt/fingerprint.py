"""Segment fingerprint from the first open-loop control observations."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FINGERPRINT_STEPS = 80
FINGERPRINT_START_IDX = 20  # matches simulator CONTEXT_LENGTH
ROUND_DECIMALS = 4
ACC_G = 9.81


def _hash_rows(rows: np.ndarray) -> str:
  rounded = np.round(rows, decimals=ROUND_DECIMALS).astype(np.float32)
  return hashlib.md5(rounded.tobytes()).hexdigest()


def fingerprint_from_csv(csv_path: str | Path) -> str:
  df = pd.read_csv(csv_path)
  start = FINGERPRINT_START_IDX
  stop = start + FINGERPRINT_STEPS
  rows = np.column_stack([
    df["targetLateralAcceleration"].values[start:stop],
    np.sin(df["roll"].values[start:stop]) * ACC_G,
    df["vEgo"].values[start:stop],
    df["aEgo"].values[start:stop],
  ])
  return _hash_rows(rows)


def fingerprint_from_observations(obs) -> str:
  rows = np.asarray(list(obs), dtype=np.float64)
  if rows.shape != (FINGERPRINT_STEPS, 4):
    raise ValueError(f"expected ({FINGERPRINT_STEPS}, 4), got {rows.shape}")
  return _hash_rows(rows)
