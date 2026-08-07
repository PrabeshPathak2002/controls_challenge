"""Gymnasium env: residual steer on top of the structured controller."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from controllers import BaseController
from controllers.structured import Controller as StructuredController
from tinyphysics import (
  CONTROL_START_IDX,
  COST_END_IDX,
  DEL_T,
  LAT_ACCEL_COST_MULTIPLIER,
  STEER_RANGE,
  TinyPhysicsModel,
  TinyPhysicsSimulator,
)

OBS_DIM = 14
FUTURE_IDXS = (0, 3, 9)


class ResidualInjector(BaseController):
  """Applies structured base + scaled residual delta in [-1, 1]."""

  def __init__(self, max_delta: float = 0.4):
    self.base = StructuredController()
    self.delta = 0.0
    self.max_delta = float(max_delta)
    self.last_base = 0.0
    self.last_applied = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.last_base = self.base.update(target_lataccel, current_lataccel, state, future_plan)
    applied = self.last_base + self.delta * self.max_delta
    self.last_applied = float(np.clip(applied, STEER_RANGE[0], STEER_RANGE[1]))
    return self.last_applied


def _pad_take(seq, idxs, default=0.0):
  out = []
  for i in idxs:
    out.append(float(seq[i]) if seq is not None and i < len(seq) else default)
  return out


def build_obs(sim: TinyPhysicsSimulator, injector: ResidualInjector) -> np.ndarray:
  target = float(sim.target_lataccel_history[-1])
  current = float(sim.current_lataccel)
  state = sim.state_history[-1]
  fp = sim.futureplan
  future_lat = fp.lataccel if fp is not None else []
  future_roll = fp.roll_lataccel if fp is not None else []
  prev_cmd = float(sim.action_history[-1]) if sim.action_history else 0.0
  error = target - current

  vals = [
    error,
    target,
    current,
    float(state.roll_lataccel),
    float(state.v_ego) / 40.0,
    float(state.a_ego),
    prev_cmd,
    injector.last_base,
    injector.last_applied,
    *_pad_take(future_lat, FUTURE_IDXS),
    *_pad_take(future_roll, FUTURE_IDXS[:2]),
  ]
  assert len(vals) == OBS_DIM
  return np.asarray(vals, dtype=np.float32)


class TinyPhysicsResidualEnv(gym.Env):
  metadata = {"render_modes": []}

  def __init__(
    self,
    model: TinyPhysicsModel,
    files: list[str],
    max_delta: float = 0.4,
    seed: int | None = None,
  ):
    super().__init__()
    self.model = model
    self.files = list(files)
    self.max_delta = max_delta
    self.np_random = np.random.default_rng(seed)

    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    self.observation_space = spaces.Box(
      low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
    )

    self.injector: ResidualInjector | None = None
    self.sim: TinyPhysicsSimulator | None = None
    self.prev_lataccel = 0.0
    self.data_path = ""

  def _make_sim(self, data_path: str) -> TinyPhysicsSimulator:
    self.injector = ResidualInjector(max_delta=self.max_delta)
    return TinyPhysicsSimulator(self.model, data_path, controller=self.injector, debug=False)

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    if seed is not None:
      self.np_random = np.random.default_rng(seed)

    self.data_path = str(self.files[int(self.np_random.integers(0, len(self.files)))])
    self.sim = self._make_sim(self.data_path)

    # Fast-forward through context + open-loop region (logged steer).
    while self.sim.step_idx < CONTROL_START_IDX:
      self.sim.step()

    self.prev_lataccel = float(self.sim.current_lataccel)
    obs = build_obs(self.sim, self.injector)
    return obs, {"file": Path(self.data_path).name}

  def step(self, action):
    assert self.sim is not None and self.injector is not None
    self.injector.delta = float(np.asarray(action).reshape(-1)[0])

    self.sim.step()
    current = float(self.sim.current_lataccel)
    target = float(self.sim.target_lataccel_history[-1])
    error = target - current
    jerk = (current - self.prev_lataccel) / DEL_T
    self.prev_lataccel = current

    # Match official total_cost ≈ 5000*mean(e²) + 100*mean(j²) over N control steps.
    n_ctrl = max(1, COST_END_IDX - CONTROL_START_IDX)
    reward = -(5000.0 / n_ctrl) * (error ** 2) - (100.0 / n_ctrl) * (jerk ** 2)
    reward -= 0.02 * (self.injector.delta ** 2)

    terminated = self.sim.step_idx >= min(COST_END_IDX, len(self.sim.data))
    truncated = False
    obs = build_obs(self.sim, self.injector) if not terminated else np.zeros(OBS_DIM, dtype=np.float32)

    info = {
      "file": Path(self.data_path).name,
      "error": error,
      "jerk": jerk,
      "step_cost": LAT_ACCEL_COST_MULTIPLIER * (error ** 2) + (jerk ** 2),
    }
    if terminated:
      total = self.sim.compute_cost()["total_cost"]
      info["total_cost"] = total
      # Light terminal shaping toward true score.
      reward -= 0.05 * float(total)
    return obs, float(reward), terminated, truncated, info


def make_model(model_path: str = "./models/tinyphysics.onnx") -> TinyPhysicsModel:
  # Keep TinyPhysics on CPU: DML/CUDA EP is slower for this tiny sequential model.
  return TinyPhysicsModel(model_path, debug=False)
