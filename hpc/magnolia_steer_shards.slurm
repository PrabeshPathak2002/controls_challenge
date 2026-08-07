#!/bin/bash
#SBATCH --job-name=steer_shards
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/steer_shard_%A_%a.out
#SBATCH --error=logs/steer_shard_%A_%a.err

# 10 shards × 500 clips = 5000 clips (indices 0..4999).
# Submit from repo root:
#   mkdir -p logs artifacts
#   sbatch hpc/magnolia_steer_shards.slurm
# Then merge:
#   python hpc/merge_steer_lookups.py --glob 'artifacts/steer_lookup_*_*.npz' --out artifacts/steer_lookup.npz

set -euo pipefail

SHARD_SIZE="${SHARD_SIZE:-500}"
WORKERS="${WORKERS:-44}"
TOTAL_CLIPS="${TOTAL_CLIPS:-5000}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs artifacts

if [[ -f .venv_magnolia/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv_magnolia/bin/activate
elif [[ -d .venv_magnolia/bin ]]; then
  export PATH="$PWD/.venv_magnolia/bin:$PATH"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

START=$(( SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
END=$(( START + SHARD_SIZE ))
if (( END > TOTAL_CLIPS )); then
  END=$TOTAL_CLIPS
fi

echo "host=$(hostname) shard=$SLURM_ARRAY_TASK_ID start=$START end=$END workers=$WORKERS"
echo "date=$(date)"

python build_steer_lookup.py \
  --start "$START" \
  --end "$END" \
  --workers "$WORKERS" \
  --data_path data \
  --model_path models/tinyphysics.onnx \
  --out "artifacts/steer_lookup_${START}_${END}.npz"

echo "done $(date)"
