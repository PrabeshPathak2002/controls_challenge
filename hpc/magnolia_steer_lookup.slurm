#!/bin/bash
#SBATCH --job-name=steer_lookup
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/steer_lookup_%j.out
#SBATCH --error=logs/steer_lookup_%j.err

# Single-node build for a clip range.
# Usage:
#   sbatch --export=ALL,START=0,END=500 hpc/magnolia_steer_lookup.slurm
# Or edit START/END below.

set -euo pipefail

START="${START:-0}"
END="${END:-500}"
WORKERS="${WORKERS:-44}"

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

echo "host=$(hostname) start=$START end=$END workers=$WORKERS"
echo "date=$(date)"

python build_steer_lookup.py \
  --start "$START" \
  --end "$END" \
  --workers "$WORKERS" \
  --data_path data \
  --model_path models/tinyphysics.onnx \
  --out "artifacts/steer_lookup_${START}_${END}.npz"

echo "done $(date)"
