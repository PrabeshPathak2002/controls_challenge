"""Per-segment CEM MPC over the future-plan window, seeded by structured."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import BaseController
from .structured import Controller as StructuredController

CONTEXT_LENGTH = 20
CONTROL_START_IDX = 100
OPEN_LOOP_UPDATES = CONTROL_START_IDX - CONTEXT_LENGTH
MAX_ACC_DELTA = 0.5
STEER_RANGE = (-2.0, 2.0)
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
VOCAB_SIZE = 1024
LATACCEL_RANGE = (-5.0, 5.0)

MODEL_CANDIDATES = [
  Path(__file__).resolve().parents[1] / "models" / "tinyphysics.onnx",
  Path("./models/tinyphysics.onnx"),
]


class _Tokenizer:
  def __init__(self):
    self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)

  def encode(self, value):
    value = np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
    return np.digitize(value, self.bins, right=True)

  def decode(self, token):
    return self.bins[token]


class _Plant:
  def __init__(self, model_path: Path):
    self.tokenizer = _Tokenizer()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    with open(model_path, "rb") as f:
      self.ort_session = ort.InferenceSession(
        f.read(), options, ["CPUExecutionProvider"]
      )

  def predict(self, states, actions, past_lats, current):
    tokenized = self.tokenizer.encode(past_lats[-CONTEXT_LENGTH:])
    raw_states = [list(x) for x in states[-CONTEXT_LENGTH:]]
    states_arr = np.column_stack(
      [actions[-CONTEXT_LENGTH:], raw_states]
    ).astype(np.float32)
    input_data = {
      "states": np.expand_dims(states_arr, axis=0),
      "tokens": np.expand_dims(tokenized, axis=0).astype(np.int64),
    }
    logits = self.ort_session.run(None, input_data)[0]
    token = int(np.argmax(logits[0, -1]))
    pred = float(self.tokenizer.decode(token))
    return float(np.clip(pred, current - MAX_ACC_DELTA, current + MAX_ACC_DELTA))


def _find_model() -> Path:
  for p in MODEL_CANDIDATES:
    if p.exists():
      return p
  raise FileNotFoundError("models/tinyphysics.onnx not found")


class Controller(BaseController):
  """
  Segment MPC: CEM over a short action sequence using future_plan as the
  reference segment. Applies the first action; shifts the plan next step.
  """

  def __init__(
    self,
    horizon=8,
    pop_size=16,
    n_elite=4,
    n_iters=2,
    replan_every=4,
    init_std=0.2,
    base_weight=8.0,
    jerk_weight=1.0,
    act_weight=0.05,
    blend=0.35,
    model_path=None,
  ):
    self.horizon = max(2, int(horizon))
    self.pop_size = int(pop_size)
    self.n_elite = int(n_elite)
    self.n_iters = int(n_iters)
    self.replan_every = max(1, int(replan_every))
    self.init_std = float(init_std)
    self.base_weight = float(base_weight)
    self.jerk_weight = float(jerk_weight)
    self.act_weight = float(act_weight)
    self.blend = float(np.clip(blend, 0.0, 1.0))

    self.base = StructuredController()
    self.plant = _Plant(Path(model_path) if model_path else _find_model())
    self.rng = np.random.default_rng(0)

    self.n_updates = 0
    self.states = []
    self.actions = []
    self.lats = []
    self.prev_action = 0.0
    self.plan = None  # remaining open-loop plan from last CEM

  def _target_at(self, target0, future_plan, h):
    if h == 0:
      return float(target0)
    if future_plan is None or not future_plan.lataccel:
      return float(target0)
    idx = min(h - 1, len(future_plan.lataccel) - 1)
    return float(future_plan.lataccel[idx])

  def _state_at(self, state0, future_plan, h):
    if h == 0:
      return state0
    if future_plan is None or not future_plan.roll_lataccel:
      return state0
    idx = min(h - 1, len(future_plan.roll_lataccel) - 1)
    return type(state0)(
      roll_lataccel=float(future_plan.roll_lataccel[idx]),
      v_ego=float(future_plan.v_ego[idx]) if future_plan.v_ego and idx < len(future_plan.v_ego) else state0.v_ego,
      a_ego=float(future_plan.a_ego[idx]) if future_plan.a_ego and idx < len(future_plan.a_ego) else state0.a_ego,
    )

  def _rollout_cost(self, seq, base_u, target0, current, state0, future_plan):
    states = list(self.states)
    actions = list(self.actions)
    past = list(self.lats)
    cur = float(current)
    prev_u = float(self.prev_action)
    prev_lat = cur
    cost = 0.0
    H = len(seq)

    for h in range(H):
      u = float(np.clip(seq[h], STEER_RANGE[0], STEER_RANGE[1]))
      st = self._state_at(state0, future_plan, h)
      if h == 0:
        # states already includes current state0 at end
        pass
      else:
        states = states + [st]
      actions_h = actions + [u]
      pred = self.plant.predict(states, actions_h, past, cur)
      tgt = self._target_at(target0, future_plan, h)
      cost += LAT_ACCEL_COST_MULTIPLIER * ((tgt - pred) ** 2)
      cost += self.jerk_weight * (((pred - prev_lat) / DEL_T) ** 2)
      cost += self.act_weight * (((u - prev_u) / DEL_T) ** 2)
      cost += self.base_weight * ((u - base_u) ** 2) * (0.85 ** h)

      actions = actions_h
      past = past + [pred]
      prev_lat = pred
      prev_u = u
      cur = pred

    return cost

  def _cem(self, base_u, target0, current, state0, future_plan):
    H = self.horizon
    if future_plan is not None and future_plan.lataccel:
      H = min(H, len(future_plan.lataccel) + 1)

    if self.plan is not None and len(self.plan) >= 1:
      mean = np.zeros(H, dtype=np.float64)
      n = min(H - 1, len(self.plan) - 1)
      if n > 0:
        mean[:n] = self.plan[1 : 1 + n]
      mean[n:] = base_u
    else:
      mean = np.full(H, base_u, dtype=np.float64)

    std = np.full(H, self.init_std, dtype=np.float64)

    best_seq = mean.copy()
    best_cost = float("inf")

    for _ in range(self.n_iters):
      samples = self.rng.normal(loc=mean, scale=std, size=(self.pop_size, H))
      samples = np.clip(samples, STEER_RANGE[0], STEER_RANGE[1])
      # Always evaluate the structured-constant and shifted-plan candidates.
      samples[0] = np.full(H, base_u)
      samples[1] = mean

      costs = np.empty(self.pop_size, dtype=np.float64)
      for i in range(self.pop_size):
        costs[i] = self._rollout_cost(samples[i], base_u, target0, current, state0, future_plan)

      elite_idx = np.argpartition(costs, self.n_elite - 1)[: self.n_elite]
      elite = samples[elite_idx]
      mean = elite.mean(axis=0)
      std = np.maximum(elite.std(axis=0), 0.03)

      i_best = int(elite_idx[np.argmin(costs[elite_idx])])
      if costs[i_best] < best_cost:
        best_cost = float(costs[i_best])
        best_seq = samples[i_best].copy()

    return best_seq

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.n_updates += 1
    current = float(current_lataccel)
    base = float(self.base.update(target_lataccel, current_lataccel, state, future_plan))
    action = float(np.clip(base, STEER_RANGE[0], STEER_RANGE[1]))

    if self.n_updates <= OPEN_LOOP_UPDATES:
      self.prev_action = action
      return action

    if self.actions and len(self.lats) == len(self.actions) - 1:
      self.lats.append(current)
    elif self.actions and len(self.lats) == len(self.actions):
      self.lats[-1] = current

    self.states.append(state)

    ready = (
      len(self.actions) >= CONTEXT_LENGTH
      and len(self.lats) >= CONTEXT_LENGTH
      and len(self.states) == len(self.actions) + 1
      and len(self.lats) == len(self.actions)
    )

    if ready:
      need_replan = (
        self.plan is None
        or len(self.plan) < 2
        or ((self.n_updates - OPEN_LOOP_UPDATES) % self.replan_every == 0)
      )
      if need_replan:
        self.plan = self._cem(base, target_lataccel, current, state, future_plan)
      planned = float(self.plan[0])
      self.plan = self.plan[1:]
      action = float(
        np.clip((1.0 - self.blend) * base + self.blend * planned, STEER_RANGE[0], STEER_RANGE[1])
      )

    self.actions.append(action)
    self.prev_action = action

    keep = CONTEXT_LENGTH * 3
    if len(self.actions) > keep:
      self.states = self.states[-keep - 1 :]
      self.actions = self.actions[-keep:]
      self.lats = self.lats[-keep:]

    return action
