"""
Focused coordinate-descent tune of controllers/structured.py.

Freezes stock PID; tunes structural knobs on a tune set with holdout checks.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = "./models/tinyphysics.onnx"
TUNE_N = 120
HOLDOUT_N = 120
WORKERS = max(1, (os.cpu_count() or 4) - 1)

# Start from preview_roll defaults + mild structural extras.
BASE = {
  "p": 0.195,
  "i": 0.100,
  "d": -0.053,
  "preview_gain": 0.4,
  "preview_steps": 4,
  "roll_gain": 0.53,
  "future_roll_gain": 0.12,
  "future_roll_steps": 4,
  "speed_k": 0.35,
  "v_ref": 20.0,
  "ff_sat": 2.5,
  "i_limit": 8.0,
}

# Only these move during search (keeps overfit risk down).
SEARCH = {
  "preview_gain": [0.30, 0.35, 0.40, 0.45, 0.50],
  "roll_gain": [0.40, 0.45, 0.50, 0.53, 0.58, 0.65],
  "future_roll_gain": [0.00, 0.05, 0.10, 0.15, 0.22, 0.30],
  "future_roll_steps": [2, 4, 6, 8],
  "speed_k": [0.00, 0.15, 0.25, 0.35, 0.50, 0.70],
  "ff_sat": [1.5, 2.0, 2.5, 3.5, 5.0],
  "i_limit": [4.0, 6.0, 8.0, 12.0, 20.0],
}

_model = None
_data_cache: dict = {}


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
  return model, model.ort_session.get_providers()


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


def init_worker(files: list[str]):
  global _model, _data_cache
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  _model, _ = make_model(MODEL_PATH)
  _data_cache = {f: _load_processed(f) for f in files}


def run_one(args):
  from controllers.structured import Controller
  from tinyphysics import TinyPhysicsSimulator

  data_path, params = args
  controller = Controller(**params)

  class CachedSim(TinyPhysicsSimulator):
    def get_data(self, data_path: str):
      return _data_cache[data_path]

  sim = CachedSim(_model, data_path, controller=controller, debug=False)
  cost = sim.rollout()
  return cost["total_cost"], cost["lataccel_cost"], cost["jerk_cost"]


def eval_params(ex, files, params):
  jobs = [(f, params) for f in files]
  chunksize = max(1, len(files) // max(1, WORKERS))
  rows = list(ex.map(run_one, jobs, chunksize=chunksize))
  totals = [r[0] for r in rows]
  lats = [r[1] for r in rows]
  jerks = [r[2] for r in rows]
  return {
    "total_cost": float(np.mean(totals)),
    "lataccel_cost": float(np.mean(lats)),
    "jerk_cost": float(np.mean(jerks)),
    "p90_total": float(np.percentile(totals, 90)),
  }


def score(metrics: dict) -> float:
  # Prefer mean, lightly penalize bad tail (leaderboard has hard clips).
  return metrics["total_cost"] + 0.15 * metrics["p90_total"]


def main():
  _, used = make_model(MODEL_PATH)
  print(f"ORT providers: {used}", flush=True)
  print(f"workers={WORKERS}", flush=True)

  all_files = [str(p) for p in sorted(Path("./data").iterdir())]
  tune_files = all_files[:TUNE_N]
  holdout_files = all_files[TUNE_N:TUNE_N + HOLDOUT_N]
  worker_files = list(dict.fromkeys(tune_files + holdout_files))
  print(f"tune={len(tune_files)} holdout={len(holdout_files)}", flush=True)

  history = []
  best = dict(BASE)
  best_tune_metrics = None
  best_hold = None

  with ProcessPoolExecutor(
    max_workers=WORKERS,
    initializer=init_worker,
    initargs=(worker_files,),
  ) as ex:
    # Baselines: structured defaults and pure preview_roll-equivalent
    preview_roll = {
      **BASE,
      "future_roll_gain": 0.0,
      "speed_k": 0.0,
      "ff_sat": 50.0,  # ~linear
      "i_limit": 1e9,
    }
    pr_tune = eval_params(ex, tune_files, preview_roll)
    pr_hold = eval_params(ex, holdout_files, preview_roll)
    print(
      f"preview_roll-like  tune={pr_tune['total_cost']:.3f}  "
      f"hold={pr_hold['total_cost']:.3f}",
      flush=True,
    )

    best_tune_metrics = eval_params(ex, tune_files, best)
    best_hold = eval_params(ex, holdout_files, best)
    print(
      f"structured start   tune={best_tune_metrics['total_cost']:.3f} "
      f"(obj={score(best_tune_metrics):.3f})  "
      f"hold={best_hold['total_cost']:.3f}",
      flush=True,
    )

    improved = True
    round_idx = 0
    while improved and round_idx < 4:
      improved = False
      round_idx += 1
      print(f"\n=== coordinate descent round {round_idx} ===", flush=True)

      for name, values in SEARCH.items():
        local_best_val = best[name]
        local_best_obj = score(best_tune_metrics)
        local_best_metrics = best_tune_metrics
        local_best_hold = best_hold

        for val in values:
          if val == best[name]:
            continue
          cand = dict(best)
          cand[name] = val
          m = eval_params(ex, tune_files, cand)
          obj = score(m)
          history.append({"round": round_idx, "param": name, "value": val, **m, "obj": obj})
          print(
            f"  try {name}={val}  tune={m['total_cost']:.3f}  "
            f"obj={obj:.3f}  p90={m['p90_total']:.3f}",
            flush=True,
          )
          if obj + 1e-6 < local_best_obj:
            # Confirm on holdout before accepting (early-stop overfit).
            h = eval_params(ex, holdout_files, cand)
            print(
              f"    holdout check: {h['total_cost']:.3f} "
              f"(best hold {local_best_hold['total_cost']:.3f})",
              flush=True,
            )
            # Accept if holdout not clearly worse.
            if h["total_cost"] <= local_best_hold["total_cost"] * 1.02:
              local_best_val = val
              local_best_obj = obj
              local_best_metrics = m
              local_best_hold = h

        if local_best_val != best[name]:
          print(
            f"  ACCEPT {name}: {best[name]} -> {local_best_val}  "
            f"tune={local_best_metrics['total_cost']:.3f}  "
            f"hold={local_best_hold['total_cost']:.3f}",
            flush=True,
          )
          best[name] = local_best_val
          best_tune_metrics = local_best_metrics
          best_hold = local_best_hold
          improved = True

  # Final head-to-head on holdout
  out = {
    "preview_roll_like": {"tune": pr_tune, "holdout": pr_hold},
    "structured_best": {
      "params": best,
      "tune": best_tune_metrics,
      "holdout": best_hold,
    },
    "holdout_delta_vs_preview_roll": pr_hold["total_cost"] - best_hold["total_cost"],
  }
  Path("structured_tune_results.json").write_text(json.dumps(out, indent=2))
  pd.DataFrame(history).to_csv("structured_tune_history.csv", index=False)
  print("\n=== DONE ===", flush=True)
  print(json.dumps(out["structured_best"], indent=2), flush=True)
  print(
    f"holdout delta (preview_roll - structured): "
    f"{out['holdout_delta_vs_preview_roll']:+.3f}",
    flush=True,
  )


if __name__ == "__main__":
  main()
