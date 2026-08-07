from . import BaseController


class Controller(BaseController):
  """PID + path preview FF + roll FF, gains meant to be CMA-ES tuned."""

  def __init__(
    self,
    p=0.195,
    i=0.100,
    d=-0.053,
    preview_gain=0.4,
    preview_steps=4,
    roll_gain=0.53,
    future_roll_gain=0.0,
  ):
    self.p = float(p)
    self.i = float(i)
    self.d = float(d)
    self.preview_gain = float(preview_gain)
    self.preview_steps = max(1, int(round(preview_steps)))
    self.roll_gain = float(roll_gain)
    self.future_roll_gain = float(future_roll_gain)
    self.error_integral = 0.0
    self.prev_error = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = target_lataccel - current_lataccel
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    pid = self.p * error + self.i * self.error_integral + self.d * error_diff

    future_lat = future_plan.lataccel if future_plan is not None else []
    if future_lat:
      idx = min(self.preview_steps - 1, len(future_lat) - 1)
      preview_target = future_lat[idx]
    else:
      preview_target = target_lataccel

    future_roll = future_plan.roll_lataccel if future_plan is not None else []
    if future_roll:
      idx = min(self.preview_steps - 1, len(future_roll) - 1)
      preview_roll = future_roll[idx]
    else:
      preview_roll = state.roll_lataccel

    preview_ff = self.preview_gain * preview_target
    roll_ff = -self.roll_gain * state.roll_lataccel
    future_roll_ff = -self.future_roll_gain * preview_roll
    return pid + preview_ff + roll_ff + future_roll_ff
