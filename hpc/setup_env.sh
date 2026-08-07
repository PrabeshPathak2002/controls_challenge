#!/bin/bash
# Setup a conda/venv env on Magnolia for steer-lookup builds.
# Run once from the repo root on Magnolia (after ssh from hpcwoods).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

module load anaconda3 2>/dev/null || true
# Magnolia may not ship an anaconda3 module; system python3 + venv is fine.
# Fallback if module name differs:
#   module avail
#   module load <python/anaconda module>

ENV_DIR="${ROOT}/.venv_magnolia"
if [[ ! -d "$ENV_DIR" ]]; then
  python3 -m venv "$ENV_DIR" || conda create -y -p "$ENV_DIR" python=3.11
fi

# shellcheck disable=SC1091
if [[ -f "$ENV_DIR/bin/activate" ]]; then
  source "$ENV_DIR/bin/activate"
else
  # conda-style prefix
  export PATH="$ENV_DIR/bin:$PATH"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# CPU onnxruntime is enough (and preferred) for TinyPhysics rollouts.
python -m pip install onnxruntime numpy pandas tqdm

python - <<'PY'
import onnxruntime as ort
print("ort", ort.__version__, ort.get_available_providers())
import tinyphysics
print("tinyphysics ok")
PY

echo "Env ready. Activate with: source ${ENV_DIR}/bin/activate"
