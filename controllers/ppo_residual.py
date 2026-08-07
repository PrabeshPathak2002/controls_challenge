from __future__ import annotations

from pathlib import Path

import numpy as np

from . import BaseController
from .structured import Controller as StructuredController

# Optional torch/SB3 — only needed when weights exist.
_POLICY = None
_POLICY_PATH = None

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "rl_runs" / "ppo_residual_v2" / "best" / "best_model.zip"
FALLBACK_MODEL = Path(__file__).resolve().parents[1] / "rl_runs" / "ppo_residual_v2" / "ppo_residual_final.zip"
FALLBACK_V1 = Path(__file__).resolve().parents[1] / "rl_runs" / "ppo_residual" / "best" / "best_model.zip"

OBS_DIM = 14
FUTURE_IDXS = (0, 3, 9)
STEER_RANGE = (-2.0, 2.0)


def _pad_take(seq, idxs, default=0.0):
  out = []
  for i in idxs:
    out.append(float(seq[i]) if seq is not None and i < len(seq) else default)
  return out


def _load_policy(path: Path):
  global _POLICY, _POLICY_PATH
  if _POLICY is not None and _POLICY_PATH == path:
    return _POLICY
  import torch
  from stable_baselines3 import PPO

  device = "cuda" if torch.cuda.is_available() else "cpu"
  _POLICY = PPO.load(str(path), device=device)
  _POLICY_PATH = path
  return _POLICY


class Controller(BaseController):
  """Structured base + PPO residual (falls back to structured if no weights)."""

  def __init__(self, max_delta=0.25, model_path=None):
    self.base = StructuredController()
    self.max_delta = float(max_delta)
    self.prev_cmd = 0.0
    self.last_base = 0.0

    if model_path:
      path = Path(model_path)
    elif DEFAULT_MODEL.exists():
      path = DEFAULT_MODEL
    elif FALLBACK_MODEL.exists():
      path = FALLBACK_MODEL
    else:
      path = FALLBACK_V1
    self.policy = None
    if path.exists():
      try:
        self.policy = _load_policy(path)
      except Exception:
        self.policy = None

  def _obs(self, target_lataccel, current_lataccel, state, future_plan):
    future_lat = future_plan.lataccel if future_plan is not None else []
    future_roll = future_plan.roll_lataccel if future_plan is not None else []
    error = target_lataccel - current_lataccel
    vals = [
      error,
      float(target_lataccel),
      float(current_lataccel),
      float(state.roll_lataccel),
      float(state.v_ego) / 40.0,
      float(state.a_ego),
      float(self.prev_cmd),
      float(self.last_base),
      float(self.prev_cmd),
      *_pad_take(future_lat, FUTURE_IDXS),
      *_pad_take(future_roll, FUTURE_IDXS[:2]),
    ]
    return np.asarray(vals, dtype=np.float32)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.last_base = self.base.update(target_lataccel, current_lataccel, state, future_plan)
    if self.policy is None:
      self.prev_cmd = self.last_base
      return self.last_base

    obs = self._obs(target_lataccel, current_lataccel, state, future_plan)
    action, _ = self.policy.predict(obs, deterministic=True)
    delta = float(np.asarray(action).reshape(-1)[0])
    cmd = self.last_base + delta * self.max_delta
    cmd = float(np.clip(cmd, STEER_RANGE[0], STEER_RANGE[1]))
    self.prev_cmd = cmd
    return cmd
