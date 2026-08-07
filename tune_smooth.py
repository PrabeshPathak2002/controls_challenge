"""Tune cmd_alpha / max_du for structured_smooth with holdout gating."""
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

BASE = {
  "p": 0.195,
  "i": 0.100,
  "d": -0.053,
  "preview_gain": 0.5,
  "preview_steps": 4,
  "roll_gain": 0.53,
  "future_roll_gain": 0.0,
  "future_roll_steps": 6,
  "speed_k": 0.0,
  "v_ref": 20.0,
  "ff_sat": 2.5,
  "i_limit": 8.0,
  "cmd_alpha": 0.55,
  "max_du": 0.35,
}

SEARCH = {
  "cmd_alpha": [0.35, 0.45, 0.55, 0.65, 0.80, 1.0],
  "max_du": [0.15, 0.25, 0.35, 0.50, 0.80, 2.0],
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


def init_worker(files: list[str]):
  global _model, _data_cache
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  _model = make_model(MODEL_PATH)
  _data_cache = {f: _load_processed(f) for f in files}


def run_one(args):
  from controllers.structured_smooth import Controller
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
  return {
    "total_cost": float(np.mean(totals)),
    "lataccel_cost": float(np.mean([r[1] for r in rows])),
    "jerk_cost": float(np.mean([r[2] for r in rows])),
    "p90_total": float(np.percentile(totals, 90)),
  }


def main():
  all_files = [str(p) for p in sorted(Path("./data").iterdir())]
  tune_files = all_files[:TUNE_N]
  holdout_files = all_files[TUNE_N:TUNE_N + HOLDOUT_N]
  worker_files = list(dict.fromkeys(tune_files + holdout_files))
  print(f"workers={WORKERS} tune={len(tune_files)} holdout={len(holdout_files)}", flush=True)

  history = []
  best = dict(BASE)

  with ProcessPoolExecutor(
    max_workers=WORKERS,
    initializer=init_worker,
    initargs=(worker_files,),
  ) as ex:
    # structured baseline = no smoothing
    structured = {**BASE, "cmd_alpha": 1.0, "max_du": 2.0}
    s_tune = eval_params(ex, tune_files, structured)
    s_hold = eval_params(ex, holdout_files, structured)
    print(
      f"structured (no smooth) tune={s_tune['total_cost']:.3f} "
      f"hold={s_hold['total_cost']:.3f} jerk={s_tune['jerk_cost']:.3f}",
      flush=True,
    )

    best_tune = eval_params(ex, tune_files, best)
    best_hold = eval_params(ex, holdout_files, best)
    print(
      f"smooth start          tune={best_tune['total_cost']:.3f} "
      f"hold={best_hold['total_cost']:.3f} jerk={best_tune['jerk_cost']:.3f}",
      flush=True,
    )

    improved = True
    round_idx = 0
    while improved and round_idx < 3:
      improved = False
      round_idx += 1
      print(f"\n=== round {round_idx} ===", flush=True)
      for name, values in SEARCH.items():
        local_val = best[name]
        local_tune = best_tune
        local_hold = best_hold
        for val in values:
          if val == best[name]:
            continue
          cand = dict(best)
          cand[name] = val
          m = eval_params(ex, tune_files, cand)
          history.append({"round": round_idx, "param": name, "value": val, **m})
          print(
            f"  {name}={val}  tune={m['total_cost']:.3f}  "
            f"jerk={m['jerk_cost']:.3f}  lat={m['lataccel_cost']:.3f}",
            flush=True,
          )
          if m["total_cost"] + 1e-6 < local_tune["total_cost"]:
            h = eval_params(ex, holdout_files, cand)
            print(
              f"    holdout={h['total_cost']:.3f} (best hold {local_hold['total_cost']:.3f})",
              flush=True,
            )
            if h["total_cost"] <= local_hold["total_cost"] * 1.02:
              local_val = val
              local_tune = m
              local_hold = h
        if local_val != best[name]:
          print(
            f"  ACCEPT {name}: {best[name]} -> {local_val}  "
            f"tune={local_tune['total_cost']:.3f} hold={local_hold['total_cost']:.3f}",
            flush=True,
          )
          best[name] = local_val
          best_tune = local_tune
          best_hold = local_hold
          improved = True

  out = {
    "structured_baseline": {"tune": s_tune, "holdout": s_hold},
    "smooth_best": {"params": best, "tune": best_tune, "holdout": best_hold},
    "holdout_delta_vs_structured": s_hold["total_cost"] - best_hold["total_cost"],
  }
  Path("smooth_tune_results.json").write_text(json.dumps(out, indent=2))
  pd.DataFrame(history).to_csv("smooth_tune_history.csv", index=False)
  print("\n=== DONE ===", flush=True)
  print(json.dumps(out["smooth_best"], indent=2), flush=True)
  print(
    f"holdout delta (structured - smooth): {out['holdout_delta_vs_structured']:+.3f}",
    flush=True,
  )


if __name__ == "__main__":
  main()
