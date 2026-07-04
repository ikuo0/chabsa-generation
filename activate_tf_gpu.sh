#!/usr/bin/env bash
set -euo pipefail

# This script is intended to be sourced so exported variables stay in the current shell.
# Usage:
#   source ./activate_tf_gpu.sh

if ! (return 0 2>/dev/null); then
  echo "[WARN] This script should be sourced: source ./activate_tf_gpu.sh"
  echo "[WARN] Continuing anyway, but environment changes will not persist in your shell."
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "/opt/venvs/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source /opt/venvs/.venv/bin/activate
    echo "[INFO] Activated virtualenv: ${VIRTUAL_ENV}"
  else
    echo "[ERROR] Virtualenv activate script not found: /opt/venvs/.venv/bin/activate" >&2
    return 1 2>/dev/null || exit 1
  fi
else
  echo "[INFO] Using existing virtualenv: ${VIRTUAL_ENV}"
fi

NVIDIA_LIB_PATHS="$(python - <<'PY'
from pathlib import Path
import site

root = Path(site.getsitepackages()[0]) / 'nvidia'
if root.exists():
    paths = [str(p) for p in sorted(root.glob('*/lib')) if p.is_dir()]
    print(':'.join(paths))
else:
    print('')
PY
)"

if [[ -z "${NVIDIA_LIB_PATHS}" ]]; then
  echo "[ERROR] Could not find pip-installed NVIDIA library directories under site-packages/nvidia" >&2
  return 1 2>/dev/null || exit 1
fi

# Prepend once while avoiding duplicates.
OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
NEW_LD_LIBRARY_PATH="${OLD_LD_LIBRARY_PATH}"
IFS=':' read -r -a LIB_ARRAY <<< "${NVIDIA_LIB_PATHS}"
for dir in "${LIB_ARRAY[@]}"; do
  [[ -z "${dir}" ]] && continue
  case ":${NEW_LD_LIBRARY_PATH}:" in
    *":${dir}:"*) ;;
    *)
      if [[ -n "${NEW_LD_LIBRARY_PATH}" ]]; then
        NEW_LD_LIBRARY_PATH="${dir}:${NEW_LD_LIBRARY_PATH}"
      else
        NEW_LD_LIBRARY_PATH="${dir}"
      fi
      ;;
  esac
done

export LD_LIBRARY_PATH="${NEW_LD_LIBRARY_PATH}"

echo "[INFO] LD_LIBRARY_PATH updated for TensorFlow GPU runtime"
python - <<'PY'
import tensorflow as tf
print('tensorflow.version=', tf.__version__)
print('is_built_with_cuda=', tf.test.is_built_with_cuda())
print('gpus=', tf.config.list_physical_devices('GPU'))
PY
