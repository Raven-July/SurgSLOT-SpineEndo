#!/usr/bin/env bash
set -euo pipefail

# Run from any directory while resolving all relative paths from the SAM2 root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
SAM2_CFG="${SAM2_CFG:-configs/surgslot/surgslot_hiera_b+_512.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/surgslot-sam2_hiera_b+.pt}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/surgslot-sam2}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

LONG_MEM="${LONG_MEM:-1}"
USE_CLS_INFERENCE="${USE_CLS_INFERENCE:-1}"
USE_CONTINUOUS_DIVEMEM="${USE_CONTINUOUS_DIVEMEM:-1}"
APPLY_POSTPROCESSING="${APPLY_POSTPROCESSING:-0}"
KEEP_VIDEO_ON_GPU="${KEEP_VIDEO_ON_GPU:-0}"
DEBUG_LONG_MEM="${DEBUG_LONG_MEM:-0}"

NUM_LONG_MEM_FRAMES="${NUM_LONG_MEM_FRAMES:-3}"
NUM_LONG_MEM_CANDIDATES="${NUM_LONG_MEM_CANDIDATES:-5}"
LONG_MEM_IOU_THR="${LONG_MEM_IOU_THR:-0.9}"
PROTO_UPDATE_THR="${PROTO_UPDATE_THR:-0.9}"
EMA_ALPHA_PROTO="${EMA_ALPHA_PROTO:-0.95}"
SHORT_MEM_FRAMES="${SHORT_MEM_FRAMES:-6}"

# Datasets used by the original SurgSLOT/SAM2 evaluation scripts.
# Set DATASETS to a space-separated subset when running only selected datasets.
DATASET_LIST=(
  "VOS-Endovis17"
  "VOS-Endovis18"
  "SurgicalAI3K"
  "polypgen"
  "Hyst-YT"
  "RARP50"
  "NUH-Hyst"
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
  echo "Model configuration not found: ${SAM2_CFG}" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Dataset root not found: ${DATA_ROOT}" >&2
  echo "Create sam2/data as a directory or symbolic link, or set DATA_ROOT." >&2
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
  --gpu_id "${GPU_ID}"
  --num_long_mem_frame "${NUM_LONG_MEM_FRAMES}"
  --num_long_mem_candidate "${NUM_LONG_MEM_CANDIDATES}"
  --long_mem_iou_thr "${LONG_MEM_IOU_THR}"
  --proto_update_thr "${PROTO_UPDATE_THR}"
  --ema_alpha_proto "${EMA_ALPHA_PROTO}"
  --short_term_frame_num "${SHORT_MEM_FRAMES}"
  --num_processes "${NUM_PROCESSES}"
  --use_all_masks
)

if [[ "${LONG_MEM}" == "1" ]]; then
  COMMON_ARGS+=(--apply_long_mem)
else
  COMMON_ARGS+=(--no-apply_long_mem)
fi
if [[ "${USE_CLS_INFERENCE}" == "1" ]]; then
  COMMON_ARGS+=(--use_cls_inference)
else
  COMMON_ARGS+=(--no-use_cls_inference)
fi
if [[ "${USE_CONTINUOUS_DIVEMEM}" == "1" ]]; then
  COMMON_ARGS+=(--use_continuous_divemem)
else
  COMMON_ARGS+=(--no-use_continuous_divemem)
fi
if [[ "${APPLY_POSTPROCESSING}" == "1" ]]; then
  COMMON_ARGS+=(--apply_postprocessing)
fi
if [[ "${KEEP_VIDEO_ON_GPU}" == "1" ]]; then
  COMMON_ARGS+=(--keep_video_on_gpu)
fi
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
    if [[ "${dataset_name}" == "VOS-Endovis18" ]]; then
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
