"""Quick holdout compare: mpc_search vs structured."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MODEL_PATH = "./models/tinyphysics.onnx"
TUNE_N = 80
HOLDOUT_N = 80
WORKERS = max(1, (os.cpu_count() or 4) - 1)

_model = None


def init_worker():
  global _model
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  from tinyphysics import TinyPhysicsModel
  _model = TinyPhysicsModel(MODEL_PATH, debug=False)


def run_one(args):
  from tinyphysics import TinyPhysicsSimulator
  import importlib
  data_path, controller_name = args
  Controller = importlib.import_module(f"controllers.{controller_name}").Controller
  sim = TinyPhysicsSimulator(_model, data_path, controller=Controller(), debug=False)
  return sim.rollout()["total_cost"]


def eval_ctrl(ex, files, name):
  jobs = [(f, name) for f in files]
  costs = list(ex.map(run_one, jobs, chunksize=max(1, len(files) // WORKERS)))
  return float(np.mean(costs)), float(np.median(costs)), float(np.percentile(costs, 90))


def main():
  files = [str(p) for p in sorted(Path("./data").iterdir())]
  tune = files[:TUNE_N]
  hold = files[TUNE_N:TUNE_N + HOLDOUT_N]
  print(f"workers={WORKERS} tune={len(tune)} holdout={len(hold)}", flush=True)
  with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker) as ex:
    for split, fl in [("tune", tune), ("holdout", hold)]:
      s_mean, s_med, s_p90 = eval_ctrl(ex, fl, "structured")
      m_mean, m_med, m_p90 = eval_ctrl(ex, fl, "mpc_search")
      print(
        f"{split}: structured mean={s_mean:.3f} med={s_med:.3f} p90={s_p90:.3f} | "
        f"mpc mean={m_mean:.3f} med={m_med:.3f} p90={m_p90:.3f} | "
        f"delta(s-m)={s_mean - m_mean:+.3f}",
        flush=True,
      )


if __name__ == "__main__":
  main()
