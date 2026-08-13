#!/usr/bin/env bash
#
# Stage a single T1w volume from a local BIDS-style RAW tree into
# docker/test/input/interf0/images/t1-brain-mri/ so a test run can
# exercise the container. The T1w file is converted to .mha (compressed)
# to match Grand Challenge's on-portal format.
#
# Usage:
#   bash docker/stage_test_input.sh <session-id>   # e.g. any session ID present under your RAW T1w tree
#
# Requires a local RAW tree containing T1w NIfTIs (BIDS-style layout was:
# <RAW_ROOT>/<subject>/<session>/anat/<subject>_<session>_..._T1w.nii.gz)
# and a Python env with SimpleITK importable.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${SCRIPT_DIR}/.." && pwd )

if [[ $# -lt 1 ]]; then
    echo "ERROR: session ID required" >&2
    echo "Usage: bash docker/stage_test_input.sh <session-id>" >&2
    exit 2
fi
SESSION_ID="${1}"
# user: set to the root of your local RAW T1w tree (BIDS-style expected).
RAW_ROOT="${RAW_ROOT:-}"
IMAGES_DIR="${SCRIPT_DIR}/test/input/interf0/images/t1-brain-mri"

# Prefer the project's local venv (SimpleITK is installed there); allow
# override via PY=/path/to/python. Fall back to system python3 only if
# neither is available.
# user: set to your project's venv python (needs SimpleITK), or leave empty
# to fall back to system python3.
_VENV_PY=""
if [[ -z "${PY:-}" ]]; then
    if [[ -x "${_VENV_PY}" ]]; then
        PY="${_VENV_PY}"
    else
        PY="python3"
    fi
fi
echo "[stage] using python: ${PY}"

if ! "${PY}" -c "import SimpleITK" 2>/dev/null; then
    echo "ERROR: SimpleITK not importable in ${PY}" >&2
    echo "       Activate the project venv or set PY=/path/to/python with SimpleITK." >&2
    exit 3
fi

if [[ ! -d "${RAW_ROOT}" ]]; then
    echo "ERROR: RAW_ROOT not found: ${RAW_ROOT}" >&2
    exit 2
fi

# BIDS-style layout: <RAW_ROOT>/<subject>/<session>/anat/<subject>_<session>_..._T1w.nii.gz
# Find any T1w volume for the requested session ID.
src=$(find "${RAW_ROOT}" -type f -name "${SESSION_ID}*T1w*.nii.gz" | head -1 || true)
if [[ -z "${src}" ]]; then
    echo "ERROR: no T1w file found for session ${SESSION_ID} under ${RAW_ROOT}" >&2
    exit 2
fi
echo "[stage] source: ${src}"

mkdir -p "${IMAGES_DIR}"
# Remove any prior staged files so the input dir has exactly one image.
rm -f "${IMAGES_DIR}"/*.mha "${IMAGES_DIR}"/*.nii.gz "${IMAGES_DIR}"/*.nii 2>/dev/null || true

dst="${IMAGES_DIR}/input.mha"
"${PY}" -c "
import SimpleITK as sitk
img = sitk.ReadImage('${src}')
sitk.WriteImage(img, '${dst}', useCompression=True)
print(f'[stage] wrote {\"${dst}\"} ({img.GetSize()}, {img.GetSpacing()})')
"

echo "[stage] done - test input ready for do_test_run.sh"
