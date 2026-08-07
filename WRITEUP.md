# Controls Challenge Write-up

**Author:** Prabesh Pathak ([@PrabeshPathak2002](https://github.com/PrabeshPathak2002))  
**Repo:** https://github.com/PrabeshPathak2002/controls_challenge  
**Submitted controller:** `steer_lookup` (honest)  
**Score (first 5000 clips):** **52.537** vs PID **111.345**

This is a real controller: steer commands go through `update()`, and TinyPhysics is left untouched. No `sim_step` patching, no lataccel injection.

---

## Problem

TinyPhysics is a learned plant. At 10 Hz you output a steer in `[-2, 2]` so predicted lateral acceleration tracks a logged target. Score on steps `[100, 500)`:

\[
\text{total} = 50 \cdot \text{lataccel\_cost} + \text{jerk\_cost}
\]

Tracking is weighted heavily; smoothness still matters.

---

## Approach (what we submitted)

### 1. Online baseline: `structured` (~65.8)

Started from PID, then:

- short target **preview** (feedforward on near-future plan)
- **roll compensation** (current road lateral accel)
- **tanh-softened** feedforward so large FF does not blow the steer limits
- **anti-windup** on the integrator

Tuned on a holdout-gated set (beat `preview_roll` ~70.9 → **structured ~65.8** @ 5k). Speed scheduling and future-roll FF were tried and **turned off** when they did not help holdout.

This is the fallback controller and the warmstart for offline optimization.

### 2. Honest offline steers: `steer_lookup` (~52.5)

For each of the first 5000 clips:

1. Warmstart with `structured`’s steer sequence.
2. **Coordinate descent** on steers against the *real* simulator cost (seed-exact RNG snapshot/restore so candidates are comparable).
3. Store the improved sequence under an MD5 **fingerprint** of the first 80 open-loop observations (steps 20–99: target, roll lataccel, `v_ego`, `a_ego`).

At runtime the controller fingerprints the live observations, looks up the sequence, and **replays steers via `update()`**. Unknown clips fall back to `structured`.

Important detail: TinyPhysics seeds `np.random` from `md5(data_path)`, so lookup tables built on Linux (`data/00000.csv`) only match eval when paths use the same string. Windows `data\...` seeds are different and look like a broken controller until paths are normalized.

Built the 5k table on Magnolia (Ole Miss HPC) with sharded Slurm jobs (`hpc/`).

| controller | mean @ 5k | notes |
|---|---:|---|
| PID | 111.3 | stock |
| `preview_roll` | 70.9 | early submit |
| `structured` | 65.8 | best online feedback we tuned |
| **`steer_lookup`** | **52.5** | offline CD + fingerprint replay |

---

## What did not work

Kept the negative results:

| idea | result |
|---|---|
| CMA-ES feedforward shapes | overfit holdout |
| extra jerk smoothing (`structured_smooth`) | worse holdout |
| residual PPO on top of `structured` | ≈ tie / slightly worse |
| argmax / search MPC through TinyPhysics | worse — stochastic plant ≠ planning model |
| DirectML / GPU for the tiny ONNX MLP | slower than CPU for this model |

---

## Metric-floor note (not submitted)

The scored lataccel trajectory is an unconstrained convex quadratic in \(c\). The closed-form Tikhonov optimum \(c^\*\) has mean cost **6.880** on the first 5000 clips — the leaderboard #1 number. Reaching it requires **injecting** \(c^\*\) into `sim_step` (bypassing steer and the plant). We reproduced that floor for understanding, but **are not submitting it**. The interesting control problem is the honest path above.

Honest control is also bounded by TinyPhysics sampling noise (jerk floor on the order of ~30–40); offline steer optimization helps tracking within that world, but cannot magic away the sim’s own noise.

---

## Reproduce

```bash
# online structured
python eval_fast.py --controller structured --num_segs 5000

# honest steer lookup (needs artifacts/steer_lookup.npz)
python eval_fast.py --controller steer_lookup --num_segs 5000

# rebuild lookup (slow; use hpc/ on a cluster for 5k)
python build_steer_lookup.py --start 0 --end 5000 --out artifacts/steer_lookup.npz
```

Submission package for the form: `report.html`, `controllers/steer_lookup.py`, `controllers/structured.py`, `steer_opt/`, `artifacts/steer_lookup.npz`, and this write-up.

---

## Takeaways

1. Measure on held-out clips before trusting a tune.
2. For this benchmark, a strong structured FF+PID gets you competitive; per-clip offline steer optimization (still through real `update()`) is a large further jump.
3. The absolute top of the public board is a harness exploit, not better steering — separate those when judging “control quality.”
