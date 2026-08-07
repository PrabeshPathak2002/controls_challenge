"""Seed-exact steer-sequence evaluator with RNG snapshot/restore."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from controllers import BaseController
from tinyphysics import (
  CONTROL_START_IDX,
  COST_END_IDX,
  DEL_T,
  LAT_ACCEL_COST_MULTIPLIER,
  STEER_RANGE,
  TinyPhysicsModel,
  TinyPhysicsSimulator,
)

OPTIMIZE_STEPS = COST_END_IDX - CONTROL_START_IDX


class _NoopController(BaseController):
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    return 0.0


class SteerEvaluator:
  """Drive the real TinyPhysicsSimulator from a fixed steer array."""

  def __init__(self, data_path: str | Path, model: TinyPhysicsModel):
    self.data_path = str(data_path)
    self.model = model
    self.steer = np.zeros(OPTIMIZE_STEPS, dtype=np.float64)
    self.sim = TinyPhysicsSimulator(
      model, self.data_path, controller=_NoopController(), debug=False
    )
    data = self.sim.data
    steer_ref = self

    def control_step(step_idx: int) -> None:
      if step_idx < CONTROL_START_IDX:
        action = data["steer_command"].values[step_idx]
      else:
        action = steer_ref.steer[step_idx - CONTROL_START_IDX]
      action = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))
      self.sim.action_history.append(action)

    self.sim.control_step = control_step

  def set_steer(self, arr) -> None:
    a = np.asarray(arr, dtype=np.float64)
    if a.shape != (OPTIMIZE_STEPS,):
      raise ValueError(f"steer must be ({OPTIMIZE_STEPS},), got {a.shape}")
    self.steer = a.copy()

  def set_steer_at(self, t_abs: int, value: float) -> None:
    self.steer[t_abs - CONTROL_START_IDX] = float(value)

  def reset(self) -> None:
    self.sim.reset()

  def advance_to(self, step_idx: int) -> None:
    while self.sim.step_idx < step_idx:
      self.sim.step()

  def _cost(self) -> dict:
    pred = np.array(self.sim.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    target = np.array(self.sim.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    lat = float(np.mean((target - pred) ** 2) * 100)
    jerk = float(np.mean((np.diff(pred) / DEL_T) ** 2) * 100)
    return {
      "lataccel_cost": lat,
      "jerk_cost": jerk,
      "total_cost": lat * LAT_ACCEL_COST_MULTIPLIER + jerk,
    }

  def roll_cost(self) -> dict:
    self.advance_to(COST_END_IDX)
    return self._cost()

  def full_cost(self, steer=None) -> dict:
    if steer is not None:
      self.set_steer(steer)
    self.reset()
    return self.roll_cost()

  def snapshot(self) -> tuple:
    sim = self.sim
    return (
      sim.step_idx,
      list(sim.state_history),
      list(sim.action_history),
      list(sim.current_lataccel_history),
      list(sim.target_lataccel_history),
      sim.current_lataccel,
      np.random.get_state(),
    )

  def restore(self, snap: tuple) -> None:
    sim = self.sim
    (sim.step_idx, sh, ah, ch, th, cl, rng) = snap
    sim.state_history = list(sh)
    sim.action_history = list(ah)
    sim.current_lataccel_history = list(ch)
    sim.target_lataccel_history = list(th)
    sim.current_lataccel = cl
    np.random.set_state(rng)

  def step(self) -> None:
    self.sim.step()
