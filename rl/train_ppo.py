"""
Train residual PPO on top of structured controller (stable-baselines3).

Example:
  python rl/train_ppo.py --timesteps 300000 --n_envs 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow `python rl/train_ppo.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--timesteps", type=int, default=1_000_000)
  p.add_argument("--n_envs", type=int, default=8)
  p.add_argument("--train_n", type=int, default=1200)
  p.add_argument("--holdout_n", type=int, default=150)
  p.add_argument("--holdout_start", type=int, default=1200)
  p.add_argument("--max_delta", type=float, default=0.25)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--out_dir", type=str, default="./rl_runs/ppo_residual_v2")
  p.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="cpu|cuda|auto — default cpu (SB3 MLP PPO is usually faster on CPU; env is ONNX/CPU-bound)",
  )
  return p.parse_args()


def resolve_device(name: str) -> str:
  import torch

  if name == "auto":
    # Tiny MLP + ONNX env: CPU is typically faster than shuttling tiny tensors to CUDA.
    return "cpu"
  if name == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
  return name


def make_env_fn(files, max_delta, seed, rank):
  def _init():
    from rl.env import TinyPhysicsResidualEnv, make_model

    model = make_model(str(ROOT / "models" / "tinyphysics.onnx"))
    env = TinyPhysicsResidualEnv(model, files, max_delta=max_delta, seed=seed + rank)
    return env

  return _init


def eval_policy(model, files, max_delta, n_episodes=None):
  from rl.env import TinyPhysicsResidualEnv, make_model

  n_episodes = n_episodes or len(files)
  phys = make_model(str(ROOT / "models" / "tinyphysics.onnx"))
  env = TinyPhysicsResidualEnv(phys, files, max_delta=max_delta, seed=0)
  costs = []
  for i in range(min(n_episodes, len(files))):
    # Force file order for comparable holdout.
    env.files = [files[i]]
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
      action, _ = model.predict(obs, deterministic=True)
      obs, _, terminated, truncated, info = env.step(action)
      done = terminated or truncated
    costs.append(info["total_cost"])
  return float(sum(costs) / len(costs)), costs


def eval_structured(files, n_episodes=None):
  from controllers.structured import Controller
  from rl.env import make_model
  from tinyphysics import TinyPhysicsSimulator

  n_episodes = n_episodes or len(files)
  phys = make_model(str(ROOT / "models" / "tinyphysics.onnx"))
  costs = []
  for i in range(min(n_episodes, len(files))):
    sim = TinyPhysicsSimulator(phys, files[i], controller=Controller(), debug=False)
    costs.append(sim.rollout()["total_cost"])
  return float(sum(costs) / len(costs))


def main():
  args = parse_args()
  import torch
  from stable_baselines3 import PPO
  from stable_baselines3.common.callbacks import EvalCallback
  from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

  device = resolve_device(args.device)
  all_files = [str(p) for p in sorted((ROOT / "data").iterdir())]
  train_files = all_files[: args.train_n]
  holdout_files = all_files[args.holdout_start : args.holdout_start + args.holdout_n]
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  print(
    f"device={device}"
    + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
    + f" train_files={len(train_files)} holdout={len(holdout_files)} "
    f"n_envs={args.n_envs} timesteps={args.timesteps}",
    flush=True,
  )

  env_fns = [
    make_env_fn(train_files, args.max_delta, args.seed, rank=i)
    for i in range(args.n_envs)
  ]
  # Subproc is faster for ONNX rollouts; Dummy is easier to debug.
  try:
    vec = SubprocVecEnv(env_fns, start_method="spawn")
  except Exception as exc:
    print(f"SubprocVecEnv failed ({exc}); falling back to DummyVecEnv", flush=True)
    vec = DummyVecEnv(env_fns)

  eval_env = DummyVecEnv([make_env_fn(holdout_files, args.max_delta, args.seed, rank=10_000)])

  model = PPO(
    "MlpPolicy",
    vec,
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    verbose=1,
    seed=args.seed,
    device=device,
  )

  eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=str(out_dir / "best"),
    log_path=str(out_dir / "eval_logs"),
    eval_freq=max(8192 // args.n_envs, 1),
    n_eval_episodes=min(30, len(holdout_files)),
    deterministic=True,
    render=False,
  )

  model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=False)
  final_path = out_dir / "ppo_residual_final.zip"
  model.save(str(final_path))
  print(f"saved {final_path}", flush=True)

  best_path = out_dir / "best" / "best_model.zip"
  load_path = best_path if best_path.exists() else final_path
  model = PPO.load(str(load_path), device=device)

  print("Evaluating structured baseline on holdout...", flush=True)
  base_mean = eval_structured(holdout_files)
  print(f"structured holdout mean: {base_mean:.3f}", flush=True)

  print("Evaluating PPO residual on holdout...", flush=True)
  ppo_mean, _ = eval_policy(model, holdout_files, args.max_delta)
  print(f"ppo_residual holdout mean: {ppo_mean:.3f}", flush=True)
  print(f"delta (structured - ppo): {base_mean - ppo_mean:+.3f}", flush=True)

  meta = {
    "timesteps": args.timesteps,
    "n_envs": args.n_envs,
    "max_delta": args.max_delta,
    "train_n": args.train_n,
    "holdout_n": args.holdout_n,
    "model_path": str(load_path),
    "structured_holdout": base_mean,
    "ppo_holdout": ppo_mean,
    "delta": base_mean - ppo_mean,
  }
  (out_dir / "results.json").write_text(json.dumps(meta, indent=2))
  print(f"wrote {out_dir / 'results.json'}", flush=True)

  vec.close()
  eval_env.close()


if __name__ == "__main__":
  main()
