"""Fast multiprocess eval of a controller on N clips; compare to a baseline CSV."""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = "./models/tinyphysics.onnx"
WORKERS = max(1, (os.cpu_count() or 4) - 1)

_model = None
_data_cache: dict = {}
_controller_name = None


def make_model(model_path: str):
  import onnxruntime as ort
  from tinyphysics import LataccelTokenizer, TinyPhysicsModel

  providers = ["CPUExecutionProvider"]
  model = TinyPhysicsModel.__new__(TinyPhysicsModel)
  model.tokenizer = LataccelTokenizer()
  options = ort.SessionOptions()
  options.intra_op_num_threads = 1
  options.inter_op_num_threads = 1
  options.log_severity_level = 3
  with open(model_path, "rb") as f:
    model.ort_session = ort.InferenceSession(f.read(), options, providers)
  return model


def _load_processed(data_path: str) -> pd.DataFrame:
  from tinyphysics import ACC_G

  df = pd.read_csv(data_path)
  return pd.DataFrame({
    "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
    "v_ego": df["vEgo"].values,
    "a_ego": df["aEgo"].values,
    "target_lataccel": df["targetLateralAcceleration"].values,
    "steer_command": -df["steerCommand"].values,
  })


def init_worker(files: list[str], controller_name: str):
  global _model, _data_cache, _controller_name
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  _model = make_model(MODEL_PATH)
  _controller_name = controller_name
  # For 5k, lazy-load in run_one to avoid huge init; cache grows over time.
  _data_cache = {}


def run_one(data_path: str):
  import importlib
  from tinyphysics import TinyPhysicsSimulator

  global _data_cache
  if data_path not in _data_cache:
    _data_cache[data_path] = _load_processed(data_path)

  Controller = importlib.import_module(f"controllers.{_controller_name}").Controller
  controller = Controller()

  class CachedSim(TinyPhysicsSimulator):
    def get_data(self, data_path: str):
      return _data_cache[data_path]

  sim = CachedSim(_model, data_path, controller=controller, debug=False)
  cost = sim.rollout()
  return {
    "file": Path(data_path).name,
    "lataccel_cost": cost["lataccel_cost"],
    "jerk_cost": cost["jerk_cost"],
    "total_cost": cost["total_cost"],
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--controller", required=True)
  parser.add_argument("--num_segs", type=int, default=5000)
  parser.add_argument("--compare_csv", default="")
  args = parser.parse_args()

  # Use posix paths so md5(data_path) seeds match Linux / Magnolia / leaderboard.
  files = [p.as_posix() for p in sorted(Path("./data").iterdir())[: args.num_segs]]
  print(f"controller={args.controller} n={len(files)} workers={WORKERS}", flush=True)

  with ProcessPoolExecutor(
    max_workers=WORKERS,
    initializer=init_worker,
    initargs=(files, args.controller),
  ) as ex:
    chunksize = max(1, len(files) // (WORKERS * 4))
    rows = list(ex.map(run_one, files, chunksize=chunksize))

  df = pd.DataFrame(rows)
  out = f"{args.controller}_{args.num_segs}_costs.csv"
  df.to_csv(out, index=False)
  print(
    f"mean total={df['total_cost'].mean():.3f}  median={df['total_cost'].median():.3f}  "
    f"p90={df['total_cost'].quantile(0.9):.3f}",
    flush=True,
  )
  print(f"SAVED {out}", flush=True)

  if args.compare_csv:
    base = pd.read_csv(args.compare_csv)
    merged = df.merge(base, on="file", suffixes=("_test", "_base"))
    d = merged["total_cost_base"].mean() - merged["total_cost_test"].mean()
    print(
      f"vs {args.compare_csv}: base_mean={merged['total_cost_base'].mean():.3f}  "
      f"test_mean={merged['total_cost_test'].mean():.3f}  delta(base-test)={d:+.3f}",
      flush=True,
    )
    wins = (merged["total_cost_test"] < merged["total_cost_base"]).mean()
    print(f"fraction clips better than baseline: {wins:.3f}", flush=True)


if __name__ == "__main__":
  main()
