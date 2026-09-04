#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time dataset download + layout for the CD-FSCIL reproduction.
#
#   bash overview/data_prepare.sh [DATA_ROOT]      # default: ./data
#
# Produces exactly the CEC/TOPIC layout every FSCIL paper uses:
#
#   data/
#   |-- miniimagenet/{images/*.jpg, split/{train,test}.csv}
#   |-- CUB_200_2011/{images/..., images.txt, train_test_split.txt, ...}
#   |-- cifar-100-python/{train,test,meta}
#   `-- index_list/{mini_imagenet,cifar100,cub200}/session_*.txt
#
# Sources
#   miniImageNet + CUB-200 : huggingface.co/datasets/HarborYuan/Few-Shot-Class-
#                            Incremental-Learning (fscil.zip, 4.2 GB).  This is
#                            the mirror the NC-FSCIL authors publish and it
#                            states it follows the CEC release; we verify below
#                            that its split CSVs are byte-identical to the ones
#                            shipped in this repo under complementary/index_list.
#   CIFAR-100              : cs.toronto.edu (official).
#
# NOTE: the miniImageNet link in the old top-level README
# (drive id 16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY) is the Ravi & Larochelle
# few-shot *cache* (mini-imagenet-cache-{train,val,test}.pkl), NOT the CEC FSCIL
# release.  It cannot reproduce Table 1 and is not used here.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${1:-data}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# the download and the self-test both need the project env
# shellcheck disable=SC1091
source "${REPO}/scripts/env.sh"
ROOT="$(mkdir -p "${ROOT}" && cd "${ROOT}" && pwd)"
DL="${ROOT}/_downloads"
mkdir -p "${DL}"

echo "==> data root: ${ROOT}"

# ---------------------------------------------------------------- fscil.zip
if [[ ! -d "${ROOT}/miniimagenet/images" || ! -d "${ROOT}/CUB_200_2011/images" ]]; then
    if [[ ! -f "${DL}/fscil.zip" ]]; then
        echo "==> downloading fscil.zip (4.2 GB)"
        python - "${DL}" <<'PY'
import sys
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="HarborYuan/Few-Shot-Class-Incremental-Learning",
                    filename="fscil.zip", repo_type="dataset",
                    local_dir=sys.argv[1])
print("saved:", p)
PY
    fi
    echo "==> extracting fscil.zip (this takes a few minutes; 72k files)"
    unzip -o -q "${DL}/fscil.zip" -d "${ROOT}"
else
    echo "==> miniimagenet/ and CUB_200_2011/ already present"
fi

# --------------------------------------------------------------- CIFAR-100
if [[ ! -d "${ROOT}/cifar-100-python" ]]; then
    if [[ ! -f "${DL}/cifar-100-python.tar.gz" ]]; then
        echo "==> downloading CIFAR-100"
        curl -L --retry 5 -C - \
            "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" \
            -o "${DL}/cifar-100-python.tar.gz"
    fi
    echo "==> extracting CIFAR-100"
    tar xzf "${DL}/cifar-100-python.tar.gz" -C "${ROOT}"
else
    echo "==> cifar-100-python/ already present"
fi

# -------------------------------------------------------------- index_list
echo "==> installing the CEC session index lists"
mkdir -p "${ROOT}/index_list"
cp -r "${REPO}/complementary/index_list/." "${ROOT}/index_list/"

# ------------------------------------------------------------ verification
echo "==> verifying the miniImageNet split matches the CEC index lists"
for f in train test; do
    a="${ROOT}/miniimagenet/split/${f}.csv"
    b="${REPO}/complementary/index_list/mini_imagenet/${f}.csv"
    if ! cmp -s "$a" "$b"; then
        echo "FATAL: ${a} differs from the CEC split ${b}." >&2
        echo "       The downloaded miniImageNet is a different variant and" >&2
        echo "       will NOT reproduce Table 1." >&2
        exit 1
    fi
    echo "    ${f}.csv  identical  ($(wc -l < "$a") lines)"
done

echo "==> protocol self-test"
cd "${REPO}"
python - "${ROOT}" <<'PY'
import json, sys
from pathlib import Path
from cdfscil.data import build_benchmark
root = sys.argv[1]
cmap = {w: v["clip_name"] for w, v in json.load(
    open(Path("cdfscil/assets/mini_imagenet_classnames.json"))).items()}
for ds in ("mini_imagenet", "cifar100", "cub200"):
    b = build_benchmark(ds, root, cmap if ds == "mini_imagenet" else None)
    info = b.sanity_check()
    print(f"  {ds:14s} OK  train={info['n_train_total']} test={info['n_test_total']} "
          f"classes={info['n_classes']} sessions={b.sessions} "
          f"base={b.base_class} way={b.way} shot={b.shot}")
print("\nDATA OK")
PY
