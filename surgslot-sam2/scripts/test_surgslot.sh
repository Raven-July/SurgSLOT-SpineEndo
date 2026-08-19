#!/usr/bin/env bash
set -euo pipefail

# Run from any directory while resolving all relative paths from the SAM2 root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAM2_CFG="${SAM2_CFG:-configs/surgslot-sam2/surgslot_hiera_b+_512.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/surgslot-sam2_hiera_b+.pt}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/surgslot-sam2}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

# GPU selection, long-term memory toggles, and all SLM thresholds
# (num_long_mem_frame, num_long_mem_candidate, long_mem_confidence_thr,
# proto_update_thr, ema_alpha_proto, short_term_frame_num) are left unset
# here because they already equal tools/vos_inference_surgslot.py's own
# defaults, which are the paper's settings. Pass extra CLI flags after
# `--` to override any of them, e.g. `bash scripts/test_surgslot.sh --gpu_id 1`.
DEBUG_LONG_MEM="${DEBUG_LONG_MEM:-0}"

# Datasets used by the original SurgSLOT/SAM2 evaluation scripts.
# Set DATASETS to a space-separated subset when running only selected datasets.
DATASET_LIST=(
  "VOS-Endovis17_demo"
  "VOS-Endovis18_demo"
#  "VOS-Endovis17"
#  "VOS-Endovis18"
#  "SurgicalAI3K"
#  "polypgen"
#  "Hyst-YT"
#  "RARP50"
#  "NUH-Hyst"
)
if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
fi

PROMPT_CONFIGS=(
#  "mask,0,mask"
#  "point,1,point_1"
  "point,3,point_3"
#  "point,5,point_5"
#  "box,0,box"
)

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  echo "Set CHECKPOINT_PATH to the released SurgSLOT-SAM2 checkpoint." >&2
  exit 1
fi

if [[ ! -f "${SAM2_CFG}" ]]; then
  # Hydra receives a package-relative config name (configs/...), while the
  # physical YAML lives below the sam2 Python package directory.
  SAM2_CFG_FILE="sam2/${SAM2_CFG}"
else
  SAM2_CFG_FILE="${SAM2_CFG}"
fi
if [[ ! -f "${SAM2_CFG_FILE}" ]]; then
  echo "Model configuration not found: ${SAM2_CFG_FILE}" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Dataset root not found: ${DATA_ROOT}" >&2
  echo "Create data as a directory or symbolic link, or set DATA_ROOT." >&2
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

COMMON_ARGS=(
  --sam2_cfg "${SAM2_CFG}"
  --sam2_checkpoint "${CHECKPOINT_PATH}"
  --num_processes "${NUM_PROCESSES}"
  --use_all_masks
)

if [[ "${DEBUG_LONG_MEM}" == "1" ]]; then
  COMMON_ARGS+=(--debug_long_mem)
fi

for prompt_config in "${PROMPT_CONFIGS[@]}"; do
  IFS=',' read -r prompt_type num_pts prompt_dir <<< "${prompt_config}"
  for dataset_name in "${DATASET_LIST[@]}"; do
    base_video_dir="${DATA_ROOT}/${dataset_name}/test/images"
    if [[ "${dataset_name}" == "RARP50" ]]; then
      input_mask_dir="${DATA_ROOT}/${dataset_name}/test/vos/Annotations_rm58"
    else
      input_mask_dir="${DATA_ROOT}/${dataset_name}/test/vos/Annotations"
    fi
    gt_root="${DATA_ROOT}/${dataset_name}/test/masks"
    output_mask_dir="${OUTPUT_DIR}/${dataset_name}/${prompt_dir}${RUN_NAME_SUFFIX}"
    benchmark_type="generic"
    # Prefix match so demo/renamed variants (e.g. VOS-Endovis18_demo) also
    # get the EndoVis18 instrument/tissue-split evaluation protocol.
    if [[ "${dataset_name}" == VOS-Endovis18* ]]; then
      benchmark_type="endovis2018"
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
      --benchmark_type "${benchmark_type}"
      --prompt_type "${prompt_type}"
      --num_pts "${num_pts}"
      "${COMMON_ARGS[@]}"
      "$@"
    )

    printf 'Running:'
    printf ' %q' "${command[@]}"
    printf '\n'
    "${command[@]}"
  done
done
