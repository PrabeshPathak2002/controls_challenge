"""
CMA-ES tune of controllers/cma_ff.py gains.

CPU multiprocess fitness (one ONNX session per worker, 1 ORT thread each).
Fitness = mean total_cost over a tune set of clips.

Speedups vs naive loop:
- whole population scored in one job batch (keeps workers saturated)
- clip CSVs preloaded once per worker
- cascade screen: cheap pass on a clip subset, full set only for survivors
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cma
import numpy as np
import pandas as pd

MODEL_PATH = "./models/tinyphysics.onnx"
TUNE_N = 80
HOLDOUT_N = 80
SCREEN_N = 20          # first-pass clips for every candidate
CMA_POP = 12
CMA_GENS = 18
WORKERS = max(1, (os.cpu_count() or 4) - 1)

# Keep anyone within this factor of the screen-best, plus at least half the pop.
SCREEN_KEEP_FRAC = 0.5
SCREEN_MARGIN = 1.12

# x = [p, i, d, preview_gain, preview_steps, roll_gain, future_roll_gain]
X0 = [0.195, 0.100, -0.053, 0.40, 4.0, 0.53, 0.10]
SIGMA0 = 0.15
BOUNDS = [
  [0.02, 0.00, -0.30, 0.00, 1.0, 0.00, 0.00],
  [0.80, 0.40, 0.05, 1.20, 20.0, 1.50, 0.80],
]

PARAM_NAMES = ["p", "i", "d", "preview_gain", "preview_steps", "roll_gain", "future_roll_gain"]

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


def x_to_params(x):
  return {
    "p": float(x[0]),
    "i": float(x[1]),
    "d": float(x[2]),
    "preview_gain": float(x[3]),
    "preview_steps": int(round(float(x[4]))),
    "roll_gain": float(x[5]),
    "future_roll_gain": float(x[6]),
  }


def run_one(args):
  from controllers.cma_ff import Controller
  from tinyphysics import TinyPhysicsSimulator

  data_path, params = args
  controller = Controller(**params)

  class CachedSim(TinyPhysicsSimulator):
    def get_data(self, data_path: str):
      return _data_cache[data_path]

  sim = CachedSim(_model, data_path, controller=controller, debug=False)
  return sim.rollout()["total_cost"]


def _mean_costs(raw: list[float], n_params: int, n_files: int) -> list[float]:
  out = []
  for i in range(n_params):
    chunk = raw[i * n_files:(i + 1) * n_files]
    out.append(float(np.mean(chunk)))
  return out


def eval_params_list(ex, files, params_list, chunksize=None):
  """Score many param dicts; submit all clip jobs in one batch."""
  if not params_list:
    return []
  n_files = len(files)
  jobs = [(f, p) for p in params_list for f in files]
  if chunksize is None:
    chunksize = max(1, n_files // max(1, WORKERS))
  raw = list(ex.map(run_one, jobs, chunksize=chunksize))
  return _mean_costs(raw, len(params_list), n_files)


def eval_params(ex, files, params):
  return eval_params_list(ex, files, [params])[0]


def cascade_scores(ex, screen_files, rest_files, params_list):
  """
  Score everyone on screen_files; finish full set only for promising candidates.
  Others keep their screen mean (pessimistic enough for CMA ranking).
  """
  n = len(params_list)
  screen = eval_params_list(ex, screen_files, params_list)
  order = np.argsort(screen)
  best_screen = float(screen[order[0]])
  keep_n = max(1, int(np.ceil(SCREEN_KEEP_FRAC * n)))
  survivors = set(int(i) for i in order[:keep_n])
  for i, s in enumerate(screen):
    if s <= best_screen * SCREEN_MARGIN:
      survivors.add(i)

  scores = list(screen)
  if rest_files and survivors:
    surv_idx = sorted(survivors)
    surv_params = [params_list[i] for i in surv_idx]
    rest_means = eval_params_list(ex, rest_files, surv_params)
    n_s, n_r = len(screen_files), len(rest_files)
    for i, rest_m in zip(surv_idx, rest_means):
      scores[i] = (n_s * screen[i] + n_r * rest_m) / (n_s + n_r)
  return scores, len(survivors)


def main():
  try:
    _, used = make_model(MODEL_PATH)
  except Exception as exc:
    raise SystemExit(
      f"Failed to load TinyPhysics ONNX session: {exc}\n"
      "Install CPU onnxruntime: python -m pip install onnxruntime"
    ) from exc

  print(f"ORT session providers: {used}", flush=True)
  print(f"workers={WORKERS} (CPU processes, 1 ORT thread each)", flush=True)

  all_files = [str(p) for p in sorted(Path("./data").iterdir())]
  tune_files = all_files[:TUNE_N]
  holdout_files = all_files[TUNE_N:TUNE_N + HOLDOUT_N]
  screen_files = tune_files[:SCREEN_N]
  rest_files = tune_files[SCREEN_N:]
  worker_files = list(dict.fromkeys(tune_files + holdout_files))

  print(
    f"tune={len(tune_files)} screen={len(screen_files)} holdout={len(holdout_files)} "
    f"pop={CMA_POP} gens={CMA_GENS}",
    flush=True,
  )

  opts = cma.CMAOptions()
  opts.set("popsize", CMA_POP)
  opts.set("maxiter", CMA_GENS)
  opts.set("bounds", BOUNDS)
  opts.set("verbose", -9)
  opts.set("seed", 42)

  es = cma.CMAEvolutionStrategy(X0, SIGMA0, opts)
  history = []

  with ProcessPoolExecutor(
    max_workers=WORKERS,
    initializer=init_worker,
    initargs=(worker_files,),
  ) as ex:
    baseline = {
      "p": 0.195,
      "i": 0.100,
      "d": -0.053,
      "preview_gain": 0.4,
      "preview_steps": 4,
      "roll_gain": 0.53,
      "future_roll_gain": 0.0,
    }
    base_score = eval_params(ex, tune_files, baseline)
    print(f"baseline preview_roll-like on tune: {base_score:.3f}", flush=True)

    gen = 0
    while not es.stop():
      gen += 1
      xs = es.ask()
      params_list = [x_to_params(x) for x in xs]
      scores, n_full = cascade_scores(ex, screen_files, rest_files, params_list)
      es.tell(xs, scores)
      best_idx = int(np.argmin(scores))
      for params, score in zip(params_list, scores):
        history.append({"gen": gen, **params, "total_cost": score})
      print(
        f"gen {gen:02d}/{CMA_GENS}  best={scores[best_idx]:.3f}  "
        f"mean={float(np.mean(scores)):.3f}  full_eval={n_full}/{len(xs)}  "
        f"params={params_list[best_idx]}",
        flush=True,
      )

    best_x = es.result.xbest
    best_params = x_to_params(best_x)
    # Re-score best on the full tune set (cascade scores can be screen-only for losers).
    best_tune = eval_params(ex, tune_files, best_params)
    print(f"\nCMA best on tune: {best_tune:.3f}  {best_params}", flush=True)

    hold_base = eval_params(ex, holdout_files, baseline)
    hold_best = eval_params(ex, holdout_files, best_params)
    print(f"holdout baseline: {hold_base:.3f}", flush=True)
    print(f"holdout cma_ff : {hold_best:.3f}", flush=True)
    print(f"holdout delta (base-cma): {hold_base - hold_best:+.3f}", flush=True)

  out = {
    "provider": used[0] if used else "CPUExecutionProvider",
    "workers": WORKERS,
    "screen_n": SCREEN_N,
    "baseline_tune": base_score,
    "best_tune": best_tune,
    "best_params": best_params,
    "holdout_baseline": hold_base,
    "holdout_cma": hold_best,
  }
  Path("cma_ff_tune_results.json").write_text(json.dumps(out, indent=2))
  pd.DataFrame(history).to_csv("cma_ff_tune_history.csv", index=False)
  print("SAVED cma_ff_tune_results.json and cma_ff_tune_history.csv", flush=True)
  print(f"CHOSEN {json.dumps(best_params)}", flush=True)


if __name__ == "__main__":
  # Windows spawn needs guard; also helps when workers import this module.
  main()
