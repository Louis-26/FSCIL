#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time environment setup for the CD-FSCIL reproduction.
#
#   bash overview/env_setup.sh
#
# Creates the conda env `FSCIL_env`, installs pinned dependencies and runs a
# self-test that fails loudly if CUDA / cuDNN / CLIP are not actually working.
# Idempotent: re-running it is safe.
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${ENV_NAME:-FSCIL_env}"
PY_VER="${PY_VER:-3.11}"

if [[ -z "${CONDA_BASE:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        cat >&2 <<'ERR'
ERROR: conda not found on PATH.

Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) or point this
script at an existing installation:

    CONDA_BASE=/path/to/miniconda3 bash overview/env_setup.sh

Everything here also works in a plain venv; the pinned package list is in the
"installing" sections below and in overview/project_setup.md.
ERR
        exit 1
    fi
    CONDA_BASE="$(conda info --base)"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found. A CUDA GPU is required for the diffusion" >&2
    echo "         stages; feature extraction and evaluation will be slow on CPU." >&2
fi

echo "==> conda base: ${CONDA_BASE}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "==> creating env ${ENV_NAME} (python ${PY_VER})"
    conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
else
    echo "==> env ${ENV_NAME} already exists"
fi
conda activate "${ENV_NAME}"

echo "==> installing PyTorch"
pip install --upgrade pip
pip install "torch==2.13.0" "torchvision==0.28.0"

echo "==> installing the rest"
pip install \
    "open_clip_torch==3.3.0" \
    "timm==1.0.29" \
    "numpy==2.4.6" \
    "pillow==12.3.0" \
    "scipy==1.17.1" \
    "pandas==3.0.5" \
    "matplotlib==3.11.1" \
    "einops==0.8.2" \
    "tqdm==4.70.0" \
    "ftfy==6.3.1" \
    "regex==2026.9.3" \
    "huggingface_hub==1.29.0" \
    "tensorboard==2.21.0"

# -------------------------------------------------------------------------
# cuDNN completeness fix.
#
# torch 2.13.0+cu130 pins nvidia-cudnn-cu13==9.20.0.48, whose wheel does NOT
# ship libcudnn_engines_tensor_ir.so.9.  If the host also has a system cuDNN
# (e.g. /usr/lib/x86_64-linux-gnu/libcudnn*.so.9 at 9.22), the dynamic loader
# fills the missing sublibrary from the system copy and every convolution dies
# with:
#     CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
#     cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
# Installing a cuDNN wheel that contains the complete 9.x sublibrary set makes
# the wheel self-consistent, so nothing is ever taken from the system.
# cuDNN 9.x is ABI-stable within the major version, so torch is happy.
# -------------------------------------------------------------------------
echo "==> pinning a complete cuDNN wheel"
pip install --upgrade "nvidia-cudnn-cu13==9.25.1.1"

echo "==> self-test"
PYV=$(python -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
NV="${CONDA_PREFIX}/lib/python${PYV}/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV}/cudnn/lib:${NV}/cublas/lib:${NV}/cuda_nvrtc/lib"

python - <<'PYTEST'
import sys, torch, torch.nn as nn
print("torch      :", torch.__version__, "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"
print("devices    :", torch.cuda.device_count(), "|", torch.cuda.get_device_name(0))
print("cudnn      :", torch.backends.cudnn.version())

# a convolution is what actually trips the cuDNN sublibrary mismatch
x = torch.randn(8, 3, 224, 224, device="cuda")
c = nn.Conv2d(3, 64, 7, stride=2, padding=3).cuda()
with torch.autocast("cuda", dtype=torch.float16):
    y = c(x)
torch.cuda.synchronize()
print("conv2d     : OK", tuple(y.shape))

# OpenAI CLIP must load with QuickGELU or every feature is silently wrong
import warnings, open_clip
with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)
    m, _, pp = open_clip.create_model_and_transforms("ViT-B-16-quickgelu",
                                                     pretrained="openai")
m = m.cuda().eval()
with torch.no_grad():
    f = m.encode_image(torch.randn(4, 3, 224, 224, device="cuda"))
assert f.shape[-1] == 512, f.shape
print("CLIP       : OK, feature dim", f.shape[-1])
print("\nENVIRONMENT OK")
PYTEST

cat <<'MSG'

==> Done.  Before running anything:

    source scripts/env.sh

(that activates the env and sets LD_LIBRARY_PATH so the complete cuDNN wheel
wins over any system copy).
MSG
