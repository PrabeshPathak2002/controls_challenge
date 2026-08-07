# Full guide: Magnolia steer-lookup build (Windows)

Goal: upload this project to MCSR Magnolia, build `artifacts/steer_lookup.npz` for 5000 clips, download it, and evaluate locally.

Your account (from MCSR welcome email):

| | |
|--|--|
| Username | `prabeshpathak` |
| **Host to use** | **`magnolia.mcsr.olemiss.edu`** |

Use **PuTTY** for the terminal and **FileZilla** for file transfer. Connect **directly to Magnolia** (not hpcwoods). Do not use the old “SSH Secure Shell” client.

---

## 0) Install tools (once)

1. **PuTTY** — https://www.putty.org/  
2. **FileZilla Client** — https://filezilla-project.org/  
   During install, decline optional adware offers.

Local files you need ready:

- `C:\Users\drago\Projects\controls_challenge\controls_challenge_magnolia_code.tar.gz`  
  (already packed; or re-run `hpc\pack_for_magnolia.ps1`)
- The challenge `data\` folder (large; all CSVs)
- Password from MCSR (change it if you ever pasted it in chat)

---

## 1) Upload with FileZilla

1. Open FileZilla.
2. Top bar:
   - **Host:** `sftp://magnolia.mcsr.olemiss.edu`
   - **Username:** `prabeshpathak`
   - **Password:** (your MCSR password)
   - **Port:** leave empty
3. Click **Quickconnect**. Accept the host key if asked.
4. Left = your PC, right = Magnolia home (`/home/prabeshpathak` or similar).

Upload the code pack:

5. Left side: go to `C:\Users\drago\Projects\controls_challenge`
6. Drag `controls_challenge_magnolia_code.tar.gz` to the right (home directory).

Upload data (big — can take a while):

7. On the right, after you extract code (step 3), enter `controls_challenge`
8. Left side: open local `data` folder
9. Drag the whole `data` folder into `controls_challenge`  
   Or upload `data` into home and move it later with PuTTY.

---

## 2) Log in with PuTTY

1. Open PuTTY.
2. **Host Name:** `magnolia.mcsr.olemiss.edu`
3. Connection type: SSH, port 22.
4. Open → login as `prabeshpathak` → enter password.
5. If this is first login, MCSR may force a password reset — do that.

You should see a prompt like:

```text
prabeshpathak@magnolia:~>
```

No second hop needed — you are already on Magnolia.

---

## 3) Unpack code on Magnolia

```bash
cd ~
mkdir -p controls_challenge
tar -xzf ~/controls_challenge_magnolia_code.tar.gz -C ~/controls_challenge
ls ~/controls_challenge
```

You should see at least:

```text
build_steer_lookup.py  controllers  hpc  models  requirements.txt  steer_opt  tinyphysics.py
```

Put data in place if you uploaded it to home:

```bash
# if data is still in home as ~/data:
mv ~/data ~/controls_challenge/data

# check a few files exist
ls ~/controls_challenge/data | head
ls ~/controls_challenge/models/tinyphysics.onnx
```

You need **~20000 CSVs** for the full dataset (eval uses first 5000).

---

## 4) One-time Python env

```bash
cd ~/controls_challenge
mkdir -p logs artifacts
bash hpc/setup_env.sh
source .venv_magnolia/bin/activate
```

If `module load anaconda3` fails inside the script:

```bash
module avail 2>&1 | less
# load whatever Anaconda/Python module Magnolia lists, e.g.:
# module load anaconda3
# then re-run:
bash hpc/setup_env.sh
source .venv_magnolia/bin/activate
```

Quick check:

```bash
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

---

## 5) Smoke test (40 clips) — do this first

Still on Magnolia, env activated:

```bash
cd ~/controls_challenge
source .venv_magnolia/bin/activate
sbatch --export=ALL,START=0,END=40,WORKERS=44 hpc/magnolia_steer_lookup.slurm
```

Watch the job:

```bash
squeue -u $USER
# replace JOBID with your id from sbatch
tail -f logs/steer_lookup_JOBID.out
```

Success looks like:

```text
SAVED artifacts/steer_lookup_0_40.npz n=40 mean structured=... -> lookup=...
```

Local reference: mean lookup ≈ **37** on 40 clips (better than structured ≈ **45**).

Cancel a stuck job if needed:

```bash
scancel JOBID
```

---

## 6) Full 5000-clip build (array of 10 shards)

```bash
cd ~/controls_challenge
source .venv_magnolia/bin/activate
sbatch hpc/magnolia_steer_shards.slurm
squeue -u $USER
```

This submits **10 jobs** (clips `0–500`, `500–1000`, … `4500–5000`) on partition **`defq`** (CPU). Each uses 44 workers. Wall time is often many hours; shards can run in parallel if the queue has free nodes.

Check progress:

```bash
squeue -u $USER
ls -lh artifacts/steer_lookup_*_*.npz
tail -f logs/steer_shard_ARRAYJOBID_0.out
```

When **all 10** `.npz` shards exist and jobs are done:

```bash
source .venv_magnolia/bin/activate
python hpc/merge_steer_lookups.py \
  --glob 'artifacts/steer_lookup_*_*.npz' \
  --out artifacts/steer_lookup.npz
ls -lh artifacts/steer_lookup.npz
```

---

## 7) Download results with FileZilla

1. FileZilla → connect to `sftp://magnolia.mcsr.olemiss.edu` again.
2. Right side: `controls_challenge/artifacts/`
3. Drag `steer_lookup.npz` to local:

`C:\Users\drago\Projects\controls_challenge\artifacts\`

(Create `artifacts` locally if missing.)

---

## 8) Evaluate at home

PowerShell:

```powershell
cd C:\Users\drago\Projects\controls_challenge
$py = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $py eval_fast.py --controller steer_lookup --num_segs 5000 --compare_csv structured_5000_costs.csv
```

`controllers/steer_lookup.py` loads `artifacts/steer_lookup.npz` by default.

Optional: regenerate official `report.html` later for resubmit if the 5k score beats `structured`.

---

## Useful Slurm cheatsheet

| Command | Meaning |
|---------|---------|
| `sbatch script.slurm` | submit |
| `squeue -u $USER` | your jobs |
| `sinfo` | queue / node status |
| `scancel JOBID` | cancel |
| `scancel -u $USER` | cancel all your jobs |
| `tail -f logs/...` | follow log |

Partition for this work: **`defq`** (CPU). Do **not** use `gpuq` for steer-lookup builds.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| PuTTY / FileZilla “access denied” | Wrong password; reset on first login; Caps Lock |
| Can’t reach `magnolia.mcsr.olemiss.edu` | Confirm VPN/campus network if required; try from PuTTY with hostname only |
| FileZilla won’t connect | Use `sftp://` prefix; not FTP |
| `module load` fails | `module avail`, load listed Anaconda, re-run setup |
| `No module named onnxruntime` | `source .venv_magnolia/bin/activate` then `pip install onnxruntime` |
| Job `PD` forever | Queue busy — wait; check `sinfo` |
| Job `OOM` / killed | Raise `#SBATCH --mem=` in the `.slurm` file |
| Lookup miss at eval | Fingerprint mismatch or incomplete merge — rebuild/merge shards |
| CRLF errors running `.sh` | Already LF-normalized in repo; if needed: `sed -i 's/\r$//' hpc/*.sh hpc/*.slurm` |

---

## Security

- Never paste your password into ChatGPT/Cursor/Discord.
- If you already did, **change it** on next login / with MCSR.
- Don’t commit passwords or `.npz` secrets into git.

---

## What “done” looks like

1. Smoke 40-clip job finished with a sensible mean.  
2. Ten shard files + merged `artifacts/steer_lookup.npz`.  
3. File on your PC under `artifacts\steer_lookup.npz`.  
4. Local 5k eval: `steer_lookup` mean **below** `structured` (~65.8).
