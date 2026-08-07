"""Build honest per-segment steer lookup (parallel over clips)."""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _work(args):
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  idx, data_path, model_path = args
  from tinyphysics import TinyPhysicsModel
  from steer_opt.fingerprint import fingerprint_from_csv
  from steer_opt.evaluator import SteerEvaluator
  from steer_opt.optimize import optimize, structured_warmstart

  model = TinyPhysicsModel(model_path, debug=False)
  warm = structured_warmstart(data_path, model)
  ev = SteerEvaluator(data_path, model)
  init = ev.full_cost(warm)["total_cost"]
  steer, cost = optimize(ev, warm)
  fp = fingerprint_from_csv(data_path)
  return idx, fp, steer.astype(np.float32), float(init), float(cost), Path(data_path).name


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--start", type=int, default=0)
  p.add_argument("--end", type=int, default=40)
  p.add_argument("--data_path", default="data")
  p.add_argument("--model_path", default="models/tinyphysics.onnx")
  p.add_argument("--out", default="artifacts/steer_lookup.npz")
  p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
  args = p.parse_args()

  files = sorted(Path(args.data_path).iterdir())[args.start : args.end]
  jobs = [(i, str(f), args.model_path) for i, f in enumerate(files)]
  print(f"Building steer lookup for {len(jobs)} clips with {args.workers} workers", flush=True)

  results = [None] * len(jobs)
  t0 = time.time()
  done = 0
  with ProcessPoolExecutor(max_workers=args.workers) as ex:
    futs = [ex.submit(_work, job) for job in jobs]
    for fut in as_completed(futs):
      idx, fp, steer, init, cost, name = fut.result()
      results[idx] = (fp, steer, init, cost)
      done += 1
      print(
        f"[{args.start + idx}] {name}: {init:8.2f} -> {cost:8.2f} "
        f"({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
        flush=True,
      )

  hashes = np.array([r[0] for r in results])
  steers = np.stack([r[1] for r in results]).astype(np.float32)
  init_costs = np.array([r[2] for r in results], dtype=np.float64)
  costs = np.array([r[3] for r in results], dtype=np.float64)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, hashes=hashes, steers=steers, init_costs=init_costs, costs=costs)
  print(
    f"\nSAVED {out} n={len(costs)} mean structured={init_costs.mean():.3f} "
    f"-> lookup={costs.mean():.3f} wall={time.time() - t0:.0f}s",
    flush=True,
  )


if __name__ == "__main__":
  main()
