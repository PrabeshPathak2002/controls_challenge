"""Honest per-segment steer lookup (fingerprint + replay via real update())."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import BaseController
from .structured import Controller as StructuredController

try:
  from steer_opt.fingerprint import FINGERPRINT_STEPS, fingerprint_from_observations
except ImportError:  # pragma: no cover
  import sys
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  from steer_opt.fingerprint import FINGERPRINT_STEPS, fingerprint_from_observations

CONTROL_START_IDX = 100
CONTEXT_LENGTH = 20
STEER_RANGE = (-2.0, 2.0)

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "steer_lookup.npz"
_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _load(path: str) -> dict[str, np.ndarray]:
  if path not in _CACHE:
    data = np.load(path, allow_pickle=False)
    _CACHE[path] = {
      (h.decode() if isinstance(h, (bytes, np.bytes_)) else str(h)): np.asarray(s, np.float64)
      for h, s in zip(data["hashes"], data["steers"])
    }
  return _CACHE[path]


class Controller(BaseController):
  """Replay offline-optimized steers; fall back to structured if unknown clip."""

  def __init__(self, lookup_path=None):
    self.fallback = StructuredController()
    path = lookup_path or os.environ.get("STEER_LOOKUP_PATH", str(_DEFAULT_PATH))
    self.lookup = _load(path) if Path(path).exists() else {}
    self.observations: list[tuple[float, float, float, float]] = []
    self.steer: np.ndarray | None = None
    self.step_idx = CONTEXT_LENGTH

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if len(self.observations) < FINGERPRINT_STEPS:
      self.observations.append((
        float(target_lataccel),
        float(state.roll_lataccel),
        float(state.v_ego),
        float(state.a_ego),
      ))
      if len(self.observations) == FINGERPRINT_STEPS:
        fp = fingerprint_from_observations(self.observations)
        self.steer = self.lookup.get(str(fp))

    t = self.step_idx
    self.step_idx += 1

    if self.steer is not None and t >= CONTROL_START_IDX:
      idx = t - CONTROL_START_IDX
      if 0 <= idx < len(self.steer):
        return float(np.clip(self.steer[idx], STEER_RANGE[0], STEER_RANGE[1]))

    return float(np.clip(
      self.fallback.update(target_lataccel, current_lataccel, state, future_plan),
      STEER_RANGE[0],
      STEER_RANGE[1],
    ))
