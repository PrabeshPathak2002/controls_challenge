"""Merge sharded steer_lookup_*.npz files into one table."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--glob", default="artifacts/steer_lookup_*_*.npz")
  p.add_argument("--out", default="artifacts/steer_lookup.npz")
  args = p.parse_args()

  files = sorted(Path().glob(args.glob))
  # Prefer numeric start order from filenames like steer_lookup_0_500.npz
  def sort_key(path: Path):
    parts = path.stem.split("_")
    try:
      return int(parts[-2])
    except Exception:
      return path.name

  files = sorted(files, key=sort_key)
  if not files:
    raise SystemExit(f"No files matched {args.glob}")

  hashes, steers, inits, costs = [], [], [], []
  for f in files:
    data = np.load(f)
    hashes.append(data["hashes"])
    steers.append(data["steers"])
    inits.append(data["init_costs"])
    costs.append(data["costs"])
    print(f"loaded {f} n={len(data['costs'])} mean={float(data['costs'].mean()):.3f}")

  out_hashes = np.concatenate(hashes)
  out_steers = np.concatenate(steers, axis=0)
  out_inits = np.concatenate(inits)
  out_costs = np.concatenate(costs)

  # Drop duplicate fingerprints (keep first).
  seen = set()
  keep = []
  for i, h in enumerate(out_hashes):
    key = h.decode() if isinstance(h, (bytes, np.bytes_)) else str(h)
    if key in seen:
      continue
    seen.add(key)
    keep.append(i)
  keep = np.asarray(keep)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    out,
    hashes=out_hashes[keep],
    steers=out_steers[keep],
    init_costs=out_inits[keep],
    costs=out_costs[keep],
  )
  print(
    f"SAVED {out} n={len(keep)} mean structured={float(out_inits[keep].mean()):.3f} "
    f"-> lookup={float(out_costs[keep].mean()):.3f}"
  )


if __name__ == "__main__":
  main()
