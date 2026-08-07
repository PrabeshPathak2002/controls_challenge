"""Short-horizon search over TinyPhysics rollouts, seeded by structured."""
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
  Conservative 1-step TinyPhysics search around structured.

  Because the real sim samples latents, the plant is imperfect — we keep a
  strong pull toward structured and only blend a fraction of the search result.
  """

  def __init__(
    self,
    horizon=1,
    deltas=(-0.15, -0.08, -0.04, 0.0, 0.04, 0.08, 0.15),
    jerk_weight=1.0,
    act_weight=0.05,
    base_weight=20.0,
    blend=0.35,
    model_path=None,
  ):
    self.horizon = max(1, int(horizon))
    self.deltas = tuple(float(x) for x in deltas)
    self.jerk_weight = float(jerk_weight)
    self.act_weight = float(act_weight)
    self.base_weight = float(base_weight)
    self.blend = float(np.clip(blend, 0.0, 1.0))
    self.base = StructuredController()
    self.plant = _Plant(Path(model_path) if model_path else _find_model())

    self.n_updates = 0
    self.states = []
    self.actions = []
    self.lats = []
    self.prev_action = 0.0

  def _future_target(self, target_lataccel, future_plan, h):
    if h == 0:
      return float(target_lataccel)
    if future_plan is None or not future_plan.lataccel:
      return float(target_lataccel)
    idx = min(h - 1, len(future_plan.lataccel) - 1)
    return float(future_plan.lataccel[idx])

  def _score(self, u0, base_u, target_lataccel, current, future_plan):
    states = list(self.states)
    actions = list(self.actions) + [float(u0)]
    past = list(self.lats)
    cur = float(current)
    prev_u = float(self.prev_action)
    prev_lat = cur
    cost = self.base_weight * ((u0 - base_u) ** 2)

    for h in range(self.horizon):
      tgt = self._future_target(target_lataccel, future_plan, h)
      u = float(u0)
      pred = self.plant.predict(states, actions, past, cur)
      cost += LAT_ACCEL_COST_MULTIPLIER * ((tgt - pred) ** 2)
      cost += self.jerk_weight * (((pred - prev_lat) / DEL_T) ** 2)
      cost += self.act_weight * (((u - prev_u) / DEL_T) ** 2)
      past = past + [pred]
      prev_lat = pred
      prev_u = u
      cur = pred
      if h + 1 < self.horizon:
        actions = actions + [u]

    return cost

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
      best_u, best_c = action, float("inf")
      for d in self.deltas:
        cand = float(np.clip(base + d, STEER_RANGE[0], STEER_RANGE[1]))
        c = self._score(cand, base, target_lataccel, current, future_plan)
        if c < best_c:
          best_c = c
          best_u = cand
      action = float(np.clip((1.0 - self.blend) * base + self.blend * best_u, STEER_RANGE[0], STEER_RANGE[1]))

    self.actions.append(action)
    self.prev_action = action

    keep = CONTEXT_LENGTH * 3
    if len(self.actions) > keep:
      self.states = self.states[-keep - 1 :]
      self.actions = self.actions[-keep:]
      self.lats = self.lats[-keep:]

    return action
