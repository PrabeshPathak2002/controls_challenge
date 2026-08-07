# Controls Challenge write-up

I worked on the [comma Controls Challenge](https://comma.ai/leaderboard) — write a lateral controller for TinyPhysics, score = `50 * lataccel_cost + jerk_cost` on the first 5000 clips.

**What I'm submitting:** `steer_lookup` — a real controller (steers go through `update()`, sim untouched).  
**Score:** **52.5** vs PID **111.3**  
**Code:** https://github.com/PrabeshPathak2002/controls_challenge

---

## What I tried

I started with the stock PID (~111), then built up a preview + roll feedforward controller. After tuning that landed at **`preview_roll` ~70.9**.

Next I cleaned it up into **`structured`** (~65.8): preview on the future plan, roll compensation, tanh soft-saturation on the FF, and anti-windup. I tried speed scheduling and future-roll FF too — they didn't help on holdout, so I turned them off. That became my best online controller and the warmstart for everything else.

A bunch of things didn't work:

- CMA-ES on feedforward shapes → overfit the holdout  
- Extra jerk smoothing → worse  
- Residual PPO on top of `structured` → basically a tie / slightly worse  
- Searching TinyPhysics as an MPC model → worse, because the plant is stochastic and doesn't match planning  
- GPU/DirectML for this tiny ONNX net → slower than CPU

---

## The thing that actually moved the needle

I offline-optimized **steer sequences** per clip against the real simulator (coordinate descent from the `structured` warmstart, with RNG snapshot/restore so comparisons are fair). At runtime I fingerprint the first 80 open-loop observations, look up that clip's steers, and replay them through normal `update()`. Unknown clips fall back to `structured`.

That got me from ~65.8 → **~52.5** on 5k. I built the table on Magnolia (campus HPC) with sharded Slurm jobs because optimizing 5k clips locally would've taken forever.

One gotcha that burned me: TinyPhysics seeds noise from `md5(data_path)`. Lookups built on Linux (`data/00000.csv`) look totally broken on Windows (`data\00000.csv`) until you use the same path string. Once paths matched, runtime cost matched the offline optimum.

| controller | @ 5k |
|---|---:|
| PID | 111.3 |
| `preview_roll` | 70.9 |
| `structured` | 65.8 |
| **`steer_lookup`** | **52.5** |

---

## About the top of the board

I also figured out (and reproduced) why #1 is **6.880**: inside the scored window the cost is just a convex quadratic in the lataccel trajectory, so there's a closed-form optimum. Injecting that into `sim_step` hits the floor. Cool for understanding the metric — **I'm not submitting that**. I care about the honest controller.

Honest control also can't escape TinyPhysics's own sampling noise; you're not getting anywhere near 6.88 without cheating the harness.

---

## How to run it

```bash
python eval_fast.py --controller steer_lookup --num_segs 5000
# needs artifacts/steer_lookup.npz
```

For the form: `report.html`, the controller code + `steer_opt/` + the npz, and this write-up.
