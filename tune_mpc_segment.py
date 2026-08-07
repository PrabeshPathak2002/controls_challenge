"""Compare mpc_segment vs structured on tune/holdout."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MODEL_PATH = "./models/tinyphysics.onnx"
TUNE_N = 40
HOLDOUT_N = 40
WORKERS = max(1, (os.cpu_count() or 4) - 1)
_model = None


def init_worker():
  global _model
  os.environ["OMP_NUM_THREADS"] = "1"
  from tinyphysics import TinyPhysicsModel
  _model = TinyPhysicsModel(MODEL_PATH, debug=False)


def run_one(args):
  import importlib
  from tinyphysics import TinyPhysicsSimulator
  path, name = args
  C = importlib.import_module(f"controllers.{name}").Controller
  return TinyPhysicsSimulator(_model, path, controller=C(), debug=False).rollout()["total_cost"]


def eval_name(ex, files, name):
  costs = list(ex.map(run_one, [(f, name) for f in files], chunksize=2))
  return float(np.mean(costs)), float(np.median(costs))


def main():
  files = [str(p) for p in sorted(Path("data").iterdir())]
  tune, hold = files[:TUNE_N], files[TUNE_N:TUNE_N + HOLDOUT_N]
  print(f"workers={WORKERS} tune={len(tune)} hold={len(hold)}", flush=True)
  with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker) as ex:
    for split, fl in [("tune", tune), ("holdout", hold)]:
      s_m, s_d = eval_name(ex, fl, "structured")
      m_m, m_d = eval_name(ex, fl, "mpc_segment")
      print(
        f"{split}: structured={s_m:.3f} (med {s_d:.3f}) | "
        f"mpc_segment={m_m:.3f} (med {m_d:.3f}) | delta(s-m)={s_m - m_m:+.3f}",
        flush=True,
      )


if __name__ == "__main__":
  main()
