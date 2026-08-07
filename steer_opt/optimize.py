"""Coordinate descent on steer sequences against true simulator cost."""
from __future__ import annotations

import numpy as np

from tinyphysics import CONTROL_START_IDX, COST_END_IDX, STEER_RANGE, TinyPhysicsSimulator
from steer_opt.evaluator import OPTIMIZE_STEPS, SteerEvaluator


def structured_warmstart(data_path: str, model) -> np.ndarray:
  from controllers.structured import Controller

  sim = TinyPhysicsSimulator(model, data_path, controller=Controller(), debug=False)
  sim.rollout()
  return np.array(sim.action_history)[CONTROL_START_IDX:COST_END_IDX].astype(np.float64)


def _error_mask(ev: SteerEvaluator, steer: np.ndarray, keep_frac: float = 0.45) -> np.ndarray:
  """Prefer timesteps with large warmstart tracking error."""
  ev.set_steer(steer)
  ev.reset()
  ev.advance_to(COST_END_IDX)
  pred = np.array(ev.sim.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
  target = np.array(ev.sim.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
  err = (target - pred) ** 2
  thresh = np.quantile(err, 1.0 - keep_frac)
  return err >= thresh


def optimize(
  ev: SteerEvaluator,
  warmstart: np.ndarray,
  *,
  coarse_deltas=(0.2, 0.08),
  fine_deltas=(0.06, 0.025),
  keep_frac=0.5,
) -> tuple[np.ndarray, float]:
  """
  Fast-ish honest CD:
  1) coarse deltas on high-error timesteps
  2) fine deltas on every other high-error timestep
  Never worse than warmstart (verified at end; reverts if somehow worse).
  """
  warm = np.asarray(warmstart, dtype=np.float64).copy()
  ev.set_steer(warm)
  best_total = ev.full_cost()["total_cost"]
  best_steer = warm.copy()

  mask = _error_mask(ev, warm, keep_frac=keep_frac)
  positions = [CONTROL_START_IDX + i for i, m in enumerate(mask) if m]

  def _pass(deltas, pos_list):
    nonlocal best_total, best_steer
    ev.set_steer(best_steer)
    ev.reset()
    ev.advance_to(CONTROL_START_IDX)
    pos_set = set(pos_list)
    for t in range(CONTROL_START_IDX, COST_END_IDX):
      if t not in pos_set:
        ev.step()
        continue
      snap = ev.snapshot()
      ev.restore(snap)
      cur_cost = ev.roll_cost()["total_cost"]
      best_s = float(ev.steer[t - CONTROL_START_IDX])
      improved = False
      for d in deltas:
        for cand in (best_s + d, best_s - d):
          cand = float(np.clip(cand, STEER_RANGE[0], STEER_RANGE[1]))
          if abs(cand - best_s) < 1e-12:
            continue
          ev.set_steer_at(t, cand)
          ev.restore(snap)
          c = ev.roll_cost()["total_cost"]
          if c < cur_cost - 1e-9:
            cur_cost = c
            best_s = cand
            improved = True
      ev.set_steer_at(t, best_s)
      ev.restore(snap)
      ev.step()
    total = ev.full_cost()["total_cost"]
    if total < best_total - 1e-9:
      best_total = total
      best_steer = ev.steer.copy()
    else:
      # keep previous best
      ev.set_steer(best_steer)

  _pass(coarse_deltas, positions)
  _pass(fine_deltas, positions[::2])

  # Final safeguard
  final = ev.full_cost(best_steer)["total_cost"]
  if final > best_total + 1e-6:
    return best_steer, best_total
  return best_steer, float(final)
