from . import BaseController
import math


class Controller(BaseController):
  """Tuned structured controller + command EMA and rate limit (jerk shaping)."""

  def __init__(
    self,
    p=0.195,
    i=0.100,
    d=-0.053,
    preview_gain=0.5,
    preview_steps=4,
    roll_gain=0.53,
    future_roll_gain=0.0,
    future_roll_steps=6,
    speed_k=0.0,
    v_ref=20.0,
    ff_sat=2.5,
    i_limit=8.0,
    steer_limit=2.0,
    # Tuned on 120 clips; holdout worse than structured — prefer structured for submit.
    # EMA: cmd = (1-alpha)*prev + alpha*raw. alpha=1 => no smoothing.
    cmd_alpha=0.65,
    # Max |Δsteer| per 0.1s step. Large => inactive.
    max_du=0.15,
  ):
    self.p = float(p)
    self.i = float(i)
    self.d = float(d)
    self.preview_gain = float(preview_gain)
    self.preview_steps = max(1, int(round(preview_steps)))
    self.roll_gain = float(roll_gain)
    self.future_roll_gain = float(future_roll_gain)
    self.future_roll_steps = max(1, int(round(future_roll_steps)))
    self.speed_k = float(speed_k)
    self.v_ref = float(v_ref)
    self.ff_sat = max(1e-3, float(ff_sat))
    self.i_limit = float(i_limit)
    self.steer_limit = float(steer_limit)
    self.cmd_alpha = min(1.0, max(0.05, float(cmd_alpha)))
    self.max_du = max(1e-3, float(max_du))

    self.error_integral = 0.0
    self.prev_error = 0.0
    self.prev_cmd = 0.0

  def _speed_scale(self, v_ego: float) -> float:
    v = max(float(v_ego), 1.0)
    scale = 1.0 + self.speed_k * (self.v_ref - v) / self.v_ref
    return min(1.8, max(0.5, scale))

  def _soft(self, x: float) -> float:
    return self.ff_sat * math.tanh(x / self.ff_sat)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = target_lataccel - current_lataccel
    error_diff = error - self.prev_error
    self.prev_error = error

    pid = self.p * error + self.i * self.error_integral + self.d * error_diff

    future_lat = future_plan.lataccel if future_plan is not None else []
    future_roll = future_plan.roll_lataccel if future_plan is not None else []

    if future_lat:
      pidx = min(self.preview_steps - 1, len(future_lat) - 1)
      preview_target = future_lat[pidx]
    else:
      preview_target = target_lataccel

    if future_roll:
      ridx = min(self.future_roll_steps - 1, len(future_roll) - 1)
      roll_ahead = future_roll[ridx]
    else:
      roll_ahead = state.roll_lataccel

    preview_ff = self.preview_gain * self._soft(preview_target)
    roll_ff = -self.roll_gain * self._soft(state.roll_lataccel)
    future_roll_ff = -self.future_roll_gain * self._soft(roll_ahead)

    speed_scale = self._speed_scale(state.v_ego)
    raw = speed_scale * (pid + preview_ff + roll_ff + future_roll_ff)

    # EMA then rate-limit (cuts command jerk → lataccel jerk)
    blended = (1.0 - self.cmd_alpha) * self.prev_cmd + self.cmd_alpha * raw
    lo = self.prev_cmd - self.max_du
    hi = self.prev_cmd + self.max_du
    command = blended if lo <= blended <= hi else (lo if blended < lo else hi)

    saturated = abs(command) >= self.steer_limit * 0.98
    same_sign = error * command > 0.0
    if not (saturated and same_sign):
      self.error_integral += error
    if self.error_integral > self.i_limit:
      self.error_integral = self.i_limit
    elif self.error_integral < -self.i_limit:
      self.error_integral = -self.i_limit

    self.prev_cmd = command
    return command
