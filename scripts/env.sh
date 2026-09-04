#!/usr/bin/env bash
# Source this before running anything:   source scripts/env.sh
# Activates the conda env and pins the CUDA library search path.
#
# Why the LD_LIBRARY_PATH line matters: some hosts ship a system cuDNN under
# /usr/lib/x86_64-linux-gnu.  If the pip cuDNN wheel is missing any sublibrary
# (torch 2.13's pinned 9.20 wheel has no libcudnn_engines_tensor_ir), the loader
# fills the gap from the system copy and every convolution aborts with
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.  overview/env_setup.sh installs a
# complete wheel; this puts it first.
#
# Override with:  CONDA_BASE=/path/to/conda ENV_NAME=myenv source scripts/env.sh

if [[ -z "${CONDA_BASE:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    elif [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
    else
        echo "scripts/env.sh: cannot find conda. Set CONDA_BASE=/path/to/conda." >&2
        return 1 2>/dev/null || exit 1
    fi
fi
ENV_NAME="${ENV_NAME:-FSCIL_env}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

PYV=$(python -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
NV="${CONDA_PREFIX}/lib/python${PYV}/site-packages/nvidia"
if [[ -d "${NV}/cudnn/lib" ]]; then
    export LD_LIBRARY_PATH="${NV}/cudnn/lib:${NV}/cublas/lib:${NV}/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

_ENV_SH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"
export PYTHONPATH="$(dirname "${_ENV_SH_DIR}")${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
unset _ENV_SH_DIR
