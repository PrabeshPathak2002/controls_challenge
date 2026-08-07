"""Build official report.html for structured vs pid using existing 5k cost CSVs + sample plots."""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import pandas as pd

from eval import SAMPLE_ROLLOUTS, create_report
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.structured import Controller as StructuredController
from controllers.pid import Controller as PidController

MODEL_PATH = "./models/tinyphysics.onnx"
TEST = "structured"
BASELINE = "pid"


def main():
  structured = pd.read_csv("structured_5000_costs.csv")
  pid = pd.read_csv("pid_5000_costs.csv")
  assert len(structured) == 5000 and len(pid) == 5000

  costs = []
  for _, row in structured.iterrows():
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
    f"means: structured={structured['total_cost'].mean():.3f}  "
    f"pid={pid['total_cost'].mean():.3f}",
    flush=True,
  )

  model = TinyPhysicsModel(MODEL_PATH, debug=False)
  files = sorted(Path("./data").iterdir())[:SAMPLE_ROLLOUTS]
  sample_rollouts = []
  print("Running sample rollouts for plots...", flush=True)
  for data_file in files:
    test_sim = TinyPhysicsSimulator(model, str(data_file), controller=StructuredController(), debug=False)
    base_sim = TinyPhysicsSimulator(model, str(data_file), controller=PidController(), debug=False)
    test_sim.rollout()
    base_sim.rollout()
    sample_rollouts.append({
      "seg": data_file.stem,
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
