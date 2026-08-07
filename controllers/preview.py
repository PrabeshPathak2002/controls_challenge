from . import BaseController


class Controller(BaseController):
  """PID plus a simple look-ahead feedforward term."""

  def __init__(self):
    # Same feedback gains as the stock PID. We only add one new idea.
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.error_integral = 0.0
    self.prev_error = 0.0

    # How strongly to steer toward the upcoming path (not just the current target).
    self.preview_gain = 0.15
    # 10 steps = 1 second ahead at 10 Hz.
    self.preview_steps = 10

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

    feedforward = self.preview_gain * preview_target
    return pid + feedforward
