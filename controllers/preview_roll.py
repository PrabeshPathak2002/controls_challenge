from . import BaseController


class Controller(BaseController):
  """Preview PID plus feedforward that counters road tilt."""

  # Tuned on clips 0-99; validated on holdout clips 100-199.
  def __init__(self, preview_gain=0.4, preview_steps=4, roll_gain=0.53):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.error_integral = 0.0
    self.prev_error = 0.0

    self.preview_gain = preview_gain
    self.preview_steps = preview_steps
    self.roll_gain = roll_gain

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = target_lataccel - current_lataccel
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    pid = self.p * error + self.i * self.error_integral + self.d * error_diff

    future = future_plan.lataccel if future_plan is not None else []
    if future:
      idx = min(self.preview_steps - 1, len(future) - 1)
      preview_target = future[idx]
    else:
      preview_target = target_lataccel

    preview_ff = self.preview_gain * preview_target
    roll_ff = -self.roll_gain * state.roll_lataccel
    return pid + preview_ff + roll_ff
