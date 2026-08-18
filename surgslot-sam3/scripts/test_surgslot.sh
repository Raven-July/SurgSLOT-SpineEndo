#!/usr/bin/env bash
set -euo pipefail

# Run from any directory while resolving all relative paths from the SAM3 root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CHECKPOINT_PATH="checkpoints/surgslot-sam3_hiera_b+.pt"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/surgslot-sam3}"
PROMPT_DIR="${PROMPT_DIR:-point_3}"

# Datasets used by the SurgSLOT/SAM2 and SurgSLOT/SAM3 evaluation scripts.
# Set DATASETS to a space-separated subset when running only selected datasets.
DATASET_LIST=(
  "VOS-Endovis17"
  "VOS-Endovis18"
#  "SurgicalAI3K"
#  "polypgen"
#  "Hyst-YT"
#  "RARP50"
#  "NUH-Hyst"
)
if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  echo "Set CHECKPOINT_PATH to the released SurgSLOT-SAM3 checkpoint." >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Dataset root not found: ${DATA_ROOT}" >&2
  echo "Create surgslot-sam3/data as a directory or symbolic link, or set DATA_ROOT." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Keep implementation details out of the result directory name. Set RUN_NAME to
# a custom label when multiple runs need to be kept side by side.
RUN_NAME="${RUN_NAME:-}"
RUN_NAME_SUFFIX=""
if [[ -n "${RUN_NAME}" ]]; then
  RUN_NAME_SUFFIX="_${RUN_NAME}"
fi

CHECKPOINT_ARGS=()
if [[ "${CHECKPOINT_PATH}" != "${DEFAULT_CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_ARGS+=(--sam3_checkpoint "${CHECKPOINT_PATH}")
fi

# Model and evaluation hyperparameters are intentionally omitted here. The
# SurgSLOT paper defaults live in vos_inference_surgslot.py and model_builder.py.
# Pass any non-default experiment options after this script's name.
for dataset_name in "${DATASET_LIST[@]}"; do
  base_video_dir="${DATA_ROOT}/${dataset_name}/test/images"
  if [[ "${dataset_name}" == "RARP50" ]]; then
    input_mask_dir="${DATA_ROOT}/${dataset_name}/test/vos/Annotations_rm58"
  else
    input_mask_dir="${DATA_ROOT}/${dataset_name}/test/vos/Annotations"
  fi
  gt_root="${DATA_ROOT}/${dataset_name}/test/masks"
  output_mask_dir="${OUTPUT_DIR}/${dataset_name}/${PROMPT_DIR}${RUN_NAME_SUFFIX}"
  benchmark_args=()
  if [[ "${dataset_name}" == "VOS-Endovis18" ]]; then
    benchmark_args+=(--benchmark_type endovis2018)
  fi

  for required_dir in "${base_video_dir}" "${input_mask_dir}" "${gt_root}"; do
    if [[ ! -d "${required_dir}" ]]; then
      echo "Required dataset directory not found: ${required_dir}" >&2
      exit 1
    fi
  done

  command=(
    "${PYTHON_BIN}"
    tools/vos_inference_surgslot.py
    --base_video_dir "${base_video_dir}"
    --input_mask_dir "${input_mask_dir}"
    --output_mask_dir "${output_mask_dir}"
    --gt_root "${gt_root}"
    "${benchmark_args[@]}"
    "${CHECKPOINT_ARGS[@]}"
    "$@"
  )

  printf 'Running:'
  printf ' %q' "${command[@]}"
  printf '\n'
  "${command[@]}"
done
