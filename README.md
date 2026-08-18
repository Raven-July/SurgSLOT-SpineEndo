# SurgSLOT: Segment Anything in Surgical Videos via Semantic Long-term Tracking

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://jinlab-imvr.github.io/SurgSLOT/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.16618-b31b1b.svg)](https://arxiv.org/abs/2511.16618)
[![License](https://img.shields.io/badge/license-multiple-blue)](#license)

</div>

**Haofeng Liu, Ziyue Wang, Sudhanshu Mishra, Mingqi Gao, Guanyi Qin, Chang Han Low, Alex Y. W. Kong, Zhu Zhuo, Huazhu Fu, Joseph S. Ng, Yueming Jin**

Official implementation of **SurgSLOT**, a promptable surgical video segmentation framework that equips both SAM 2 and SAM 3 with semantic long-term tracking.

> [SurgSLOT: Segment Anything in Surgical Videos via Semantic Long-term Tracking](https://arxiv.org/abs/2511.16618)

## Overview

Interactive video object segmentation can generalize to new surgical centers, procedures, and object categories without retraining: a user specifies the target with a first-frame visual prompt, and the model tracks that target throughout the video. Long surgical videos remain difficult, however, because instruments and tissues may disappear for extended periods and later reappear near visually similar distractors.

SurgSLOT addresses this problem through a stable object-level semantic identity shared by two coupled modules:

- **Temporal Semantic Learning (TSL)** decodes a semantic token for each frame and aggregates reliable target appearances into a slowly evolving semantic anchor. Temporal coherence and vision-language supervision make the identity discriminative enough to re-identify a target after long absences.
- **Semantic-driven Long-term Memory (SLM)** trains memory attention with long-range references and selects inference-time memories by their agreement with the semantic anchor, reducing identity drift from unreliable or visually similar frames.

SurgSLOT is developed together with **iSurg**, a generalization-oriented surgical segmentation benchmark that unifies 16 datasets across six procedure types.

<p align="center">
  <img src="https://jinlab-imvr.github.io/SurgSLOT/static/images/intro.png" width="95%" alt="Overview of the iSurg benchmark and SurgSLOT">
</p>

## Highlights

- **Two backbone variants:** SurgSLOT-SAM2 and SurgSLOT-SAM3.
- **Large-scale surgical benchmark:** 735 videos, 176,357 frames, 414,379 object masks, and 2,450 masklets across six procedure types.
- **Cross-dataset generalization:** every evaluation source is held out from training.
- **Zero-shot transfer:** evaluated on an unseen procedure type and unseen object categories.
- **Long-duration tracking:** robust recovery after prolonged target disappearance in videos up to 30 minutes.
- **Real-time SAM2 inference:** 68 FPS on an NVIDIA RTX A6000.

## Method

<p align="center">
  <img src="https://jinlab-imvr.github.io/SurgSLOT/static/images/architecture.png" width="95%" alt="SurgSLOT architecture">
</p>

The semantic anchor connects learning and memory selection. TSL learns an object identity that remains stable across time, while SLM uses the same identity to retain reliable long-term references. This coupling allows the tracker to recover the prompted target instead of drifting to a similar-looking object.

## Performance

### Cross-dataset generalization

Macro Average J&F with three-click initialization:

| Model | Backbone | Macro Average J&F | Gain over fine-tuned backbone |
|---|---|---:|---:|
| SurgSLOT-SAM2 | SAM 2 | 81.0 | +5.1 |
| SurgSLOT-SAM3 | SAM 3 | 82.8 | +5.3 |

### Zero-shot procedure transfer

Nephrectomy is absent from training. Results below report J&F with three-click initialization.

| Model | EndoVis17 | EndoVis18 Instrument | EndoVis18 Tissue |
|---|---:|---:|---:|
| SurgSLOT-SAM2 | 86.6 | 85.3 | 74.8 |
| SurgSLOT-SAM3 | 87.6 | 90.9 | 76.2 |

Please see the [paper](https://arxiv.org/abs/2511.16618) and [project page](https://jinlab-imvr.github.io/SurgSLOT/) for complete per-dataset results, long-term tracking analyses, and video demonstrations.

## Repository Structure

```text
SurgSLOT/
├── sam2/                 # SurgSLOT-SAM2, based on SAM 2.1
├── surgslot-sam3/        # SurgSLOT-SAM3, based on SAM 3
├── LICENSE               # Apache-2.0 for original SurgSLOT materials
├── NOTICE                # Directory-level and third-party license notice
└── README.md
```

SAM 2 and SAM 3 require different Python and PyTorch environments. Install and use each backbone from its own directory.

## Release Status

The repository is being prepared for a reproducible public release. Links will be activated as each artifact is finalized.

| Artifact | Status |
|---|---|
| SurgSLOT-SAM2 model code | Available in `sam2/` |
| SurgSLOT-SAM2 checkpoint | Release in progress |
| SurgSLOT-SAM2 benchmark script | Available at `sam2/scripts/test_surgslot.sh` |
| SurgSLOT-SAM2 training configuration | Available under `sam2/sam2/configs/surgslot/` |
| SurgSLOT-SAM3 model code and checkpoint | Release in progress |

## Installation

### SurgSLOT-SAM2

The SAM2 implementation is based on Python 3.10, PyTorch 2.5.1, and CUDA 12.4. A Linux server with an NVIDIA GPU is recommended. Install the CUDA toolkit (including `nvcc`) matching the CUDA version used by PyTorch if you want to build SAM2's optional CUDA post-processing extension.

```bash
git clone https://github.com/jinlab-imvr/SurgSLOT.git
cd SurgSLOT/sam2

conda create -n surgslot-sam2 python=3.10 -y
conda activate surgslot-sam2

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]" --no-build-isolation -v
pip install -r sav_dataset/requirements.txt
```

The `dev` extra is required for SurgSLOT training because it installs training dependencies including `transformers`, `submitit`, and TensorBoard. The evaluator dependencies are installed separately from `sav_dataset/requirements.txt`.

If the CUDA extension cannot be compiled, install without it (mask hole-filling/post-processing will be unavailable, but the main image and video model remains usable):

```bash
SAM2_BUILD_CUDA=0 pip install -e ".[dev]" --no-build-isolation -v
```

Verify that the editable installation resolves to this checkout:

```bash
python -c "import sam2; print(sam2.__file__)"
python training/train.py --help
python tools/vos_inference_surgslot.py --help
```

See [SAM 2 installation notes](sam2/INSTALL.md) for CUDA-extension troubleshooting and alternative installation options. Training initializes its semantic text encoder from `openai/clip-vit-base-patch32`; on an offline compute server, download/cache that Hugging Face model before launching training and expose the cache through `HF_HOME` (or the standard Hugging Face cache location).

### SurgSLOT-SAM3

SAM 3 has a separate environment requirement. Follow the prerequisites and installation instructions in [the SAM3 README](surgslot-sam3/README.md) from inside `surgslot-sam3/`. SurgSLOT-SAM3-specific commands and checkpoints will be added with its release.

## Checkpoints

Large model weights are not committed to Git. SurgSLOT checkpoint download links and SHA-256 checksums will be provided here when the release artifacts are finalized.

| Model | Backbone | Configuration | Checkpoint |
|---|---|---|---|
| SurgSLOT-SAM2 | SAM 2.1 Hiera B+ | [`surgslot_hiera_b+_512.yaml`](sam2/sam2/configs/surgslot/surgslot_hiera_b+_512.yaml) | Coming soon |
| SurgSLOT-SAM3 | SAM 3 | Coming soon | Coming soon |

Place the SAM2 checkpoint under `sam2/checkpoints/` or pass an absolute path to the inference command.

Training starts from Meta's official `sam2.1_hiera_base_plus.pt` backbone checkpoint. From `SurgSLOT/sam2`, either run `bash checkpoints/download_ckpts.sh` (downloads all four SAM 2.1 checkpoints) or download only the Base+ checkpoint from the [official SAM 2.1 checkpoint directory](https://github.com/facebookresearch/sam2/tree/main/checkpoints). The training configurations expect it at:

```text
sam2/checkpoints/sam2.1_hiera_base_plus.pt
```

## Data Preparation

The benchmark code expects the following layout for each evaluation dataset:

```text
sam2/data/<DATASET>/test/
├── images/
│   └── <VIDEO_0>/
│       ├── 00000.jpg
│       └── ...
├── vos/
│   └── Annotations/
│       └── <VIDEO_0>/
│           ├── 00000.png
│           └── ...
└── masks/
    └── <VIDEO_0>/
        ├── 00000.png
        └── ...
```

The datasets are not redistributed by this repository. Please obtain each dataset from its official provider and comply with its license and access terms. Existing data can be linked instead of copied:

```bash
cd SurgSLOT/sam2
ln -s /path/to/isurg/data data
```

## SAM2 Inference

Run the following command from `SurgSLOT/sam2`. The SurgSLOT inference configuration defaults to the paper setting: long-term memory enabled with capacity 3, a five-frame candidate buffer, confidence threshold 0.9, EMA decay 0.95, and six short-term memory frames. This example uses a three-point prompt.

```bash
python tools/vos_inference_surgslot.py \
  --sam2_checkpoint checkpoints/surgslot-sam2_hiera_b+.pt \
  --sam2_cfg configs/surgslot/surgslot_hiera_b+_512.yaml \
  --base_video_dir ./data/VOS-Endovis17/test/images \
  --input_mask_dir ./data/VOS-Endovis17/test/vos/Annotations \
  --output_mask_dir ./outputs/VOS-Endovis17/point_3 \
  --gt_root ./data/VOS-Endovis17/test/masks \
  --gpu_id 0 \
  --prompt_type point \
  --num_pts 3 \
  --apply_long_mem \
  --use_cls_inference \
  --use_continuous_divemem \
  --num_long_mem_frame 3 \
  --num_long_mem_candidate 5 \
  --long_mem_iou_thr 0.9 \
  --proto_update_thr 0.9 \
  --ema_alpha_proto 0.95 \
  --short_term_frame_num 6
```

To run all seven evaluation datasets and all five prompt settings (mask, 1/3/5 points, and box), use the benchmark script from `SurgSLOT/sam2`:

```bash
CHECKPOINT_PATH=/path/to/surgslot-sam2_hiera_b+.pt \
DATA_ROOT=/path/to/isurg/data \
OUTPUT_DIR=./outputs/surgslot-sam2 \
GPU_ID=0 \
bash scripts/test_surgslot.sh
```

The defaults follow the paper's long-term memory settings. Environment variables can override them without editing the script. For example, test only two datasets with four CPU evaluation workers:

```bash
DATASETS="VOS-Endovis17 VOS-Endovis18" \
NUM_PROCESSES=4 \
CHECKPOINT_PATH=/path/to/surgslot-sam2_hiera_b+.pt \
bash scripts/test_surgslot.sh
```

Before a server run, syntax-check the shell script with `bash -n scripts/test_surgslot.sh`. The script validates the checkpoint, configuration, and expected dataset directories before starting inference.

## Training

SurgSLOT-SAM2 training uses the SAM 2 training framework together with Temporal Semantic Learning losses and long-range memory sampling. The released configuration uses 512-pixel inputs, 10-frame video samples, and four GPUs by default.

Before training:

1. Link or place the training datasets under `SurgSLOT/sam2/data` using the paths declared in [`surgslot_train_512.yaml`](sam2/sam2/configs/surgslot/datasets/surgslot_train_512.yaml).
2. Place Meta's `sam2.1_hiera_base_plus.pt` at `SurgSLOT/sam2/checkpoints/sam2.1_hiera_base_plus.pt`.
3. Ensure `openai/clip-vit-base-patch32` is available from Hugging Face or already present in the server cache.

Launch locally with four GPUs from `SurgSLOT/sam2`:

```bash
python training/train.py \
  -c configs/surgslot/surgslot-sam2_hiera_b+_training_512_compact.yaml \
  --use-cluster 0 \
  --num-gpus 4
```

The compact configuration composes the dataset block from a separate file and is recommended for maintenance. The self-contained equivalent is [`surgslot-sam2_hiera_b+_training_512.yaml`](sam2/sam2/configs/surgslot/surgslot-sam2_hiera_b+_training_512.yaml). Override `--num-gpus` to match the server; effective batch size and memory use change accordingly. Logs and checkpoints are written below `sam2_logs/<config-name>/` unless `launcher.experiment_log_dir` is changed.

## Acknowledgements

This work builds on [SAM 2](https://github.com/facebookresearch/sam2), [SAM 3](https://github.com/facebookresearch/sam3), and [CLIP](https://github.com/openai/CLIP). We thank their authors for making these resources available.

iSurg consolidates data from multiple surgical datasets. We thank the dataset creators and National University Hospital for supporting evaluation on in-house hysterectomy videos. Users must request or download datasets from their respective official sources.

The SAM2 evaluation utilities include code adapted from the DAVIS evaluation and VOS benchmark projects; their original license files are retained under `sam2/sav_dataset/`.

## License

This is a **multi-license repository**:

- Original SurgSLOT code and documentation are released under the [Apache License 2.0](LICENSE), unless a more specific license applies in a subdirectory.
- The `sam2/` directory is based on Meta's SAM 2 and remains subject to its [Apache License 2.0](sam2/LICENSE). Additional third-party terms are retained in the [cc_torch license](sam2/LICENSE_cctorch), [SA-V dataset license](sam2/sav_dataset/LICENSE), [DAVIS license](sam2/sav_dataset/LICENSE_DAVIS), and [VOS benchmark license](sam2/sav_dataset/LICENSE_VOS_BENCHMARK).
- The `surgslot-sam3/` directory contains SAM 3 materials and derivative modifications. It is governed by the [SAM License](surgslot-sam3/LICENSE), not by the root Apache-2.0 license. By using or redistributing that directory, you agree to the SAM License and must provide a copy of it with the materials.
- Model checkpoints may additionally be subject to the terms of their underlying backbone and their release host.

See [NOTICE](NOTICE) for the directory-level license map and attribution summary. This repository is provided for research purposes and is not a clinically validated medical device.

## Citation

If you find SurgSLOT useful, please cite:

```bibtex
@article{liu2025surgslot,
  title   = {SurgSLOT: Segment Anything in Surgical Videos via Semantic Long-term Tracking},
  author  = {Haofeng Liu and Ziyue Wang and Sudhanshu Mishra and Mingqi Gao and
             Guanyi Qin and Chang Han Low and Alex Y. W. Kong and Zhu Zhuo and
             Huazhu Fu and Joseph S. Ng and Yueming Jin},
  journal = {arXiv preprint arXiv:2511.16618},
  year    = {2025}
}
```
