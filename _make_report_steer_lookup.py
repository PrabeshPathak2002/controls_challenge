"""Build official report.html for steer_lookup vs pid using existing 5k cost CSVs + sample plots."""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import pandas as pd

from eval import SAMPLE_ROLLOUTS, create_report
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.steer_lookup import Controller as SteerLookupController
from controllers.pid import Controller as PidController

MODEL_PATH = "./models/tinyphysics.onnx"
TEST = "steer_lookup"
BASELINE = "pid"


def main():
  test = pd.read_csv("steer_lookup_5000_costs.csv")
  pid = pd.read_csv("pid_5000_costs.csv")
  assert len(test) == 5000 and len(pid) == 5000

  costs = []
  for _, row in test.iterrows():
    costs.append({
      "controller": "test",
      "lataccel_cost": float(row["lataccel_cost"]),
      "jerk_cost": float(row["jerk_cost"]),
      "total_cost": float(row["total_cost"]),
    })
  for _, row in pid.iterrows():
    costs.append({
      "controller": "baseline",
      "lataccel_cost": float(row["lataccel_cost"]),
      "jerk_cost": float(row["jerk_cost"]),
      "total_cost": float(row["total_cost"]),
    })

  print(
    f"means: steer_lookup={test['total_cost'].mean():.3f}  "
    f"pid={pid['total_cost'].mean():.3f}",
    flush=True,
  )

  model = TinyPhysicsModel(MODEL_PATH, debug=False)
  # posix paths so md5 seeds match Magnolia / Linux leaderboard
  files = [p.as_posix() for p in sorted(Path("./data").iterdir())[:SAMPLE_ROLLOUTS]]
  sample_rollouts = []
  print("Running sample rollouts for plots...", flush=True)
  for data_file in files:
    test_sim = TinyPhysicsSimulator(model, data_file, controller=SteerLookupController(), debug=False)
    base_sim = TinyPhysicsSimulator(model, data_file, controller=PidController(), debug=False)
    test_sim.rollout()
    base_sim.rollout()
    sample_rollouts.append({
      "seg": Path(data_file).stem,
      "test_controller": TEST,
      "baseline_controller": BASELINE,
      "desired_lataccel": test_sim.target_lataccel_history,
      "test_controller_lataccel": test_sim.current_lataccel_history,
      "baseline_controller_lataccel": base_sim.current_lataccel_history,
    })

  create_report(TEST, BASELINE, sample_rollouts, costs, 5000)
  print("Done.", flush=True)


if __name__ == "__main__":
  main()
