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
├── surgslot-sam2/        # SurgSLOT-SAM2, based on SAM 2.1
├── surgslot-sam3/        # SurgSLOT-SAM3, based on SAM 3
├── LICENSE               # Apache-2.0 for original SurgSLOT materials
├── NOTICE                # Directory-level and third-party license notice
└── README.md
```

SAM 2 and SAM 3 require different Python and PyTorch environments. Install and use each backbone from its own directory.

## Installation

### SurgSLOT-SAM2

The SAM2 implementation is based on Python 3.10, PyTorch 2.5.1, and CUDA 12.4. A Linux server with an NVIDIA GPU is recommended. Install the CUDA toolkit (including `nvcc`) matching the CUDA version used by PyTorch if you want to build SAM2's optional CUDA post-processing extension. Create a dedicated Conda environment for SAM2 — its Python/PyTorch/CUDA versions differ from SurgSLOT-SAM3's, so the two must not share an environment.

```bash
git clone https://github.com/jinlab-imvr/SurgSLOT.git
cd SurgSLOT/surgslot-sam2

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

See [SAM 2 installation notes](surgslot-sam2/INSTALL.md) for CUDA-extension troubleshooting and alternative installation options. Training initializes its semantic text encoder from `openai/clip-vit-base-patch32`; on an offline compute server, download/cache that Hugging Face model before launching training and expose the cache through `HF_HOME` (or the standard Hugging Face cache location).

### SurgSLOT-SAM3

The SAM3 implementation requires Python 3.12+, PyTorch 2.7+, and CUDA 12.6+ — different versions from SurgSLOT-SAM2. Create a **separate** Conda environment for it; do not reuse the `surgslot-sam2` environment.

```bash
cd SurgSLOT/surgslot-sam3

conda create -n surgslot-sam3 python=3.12 -y
conda activate surgslot-sam3

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train,dev]"
pip install -r sav_dataset/requirements.txt
```

There is no compiled CUDA extension to build here (unlike SAM2's `setup.py`) — `pip install -e .` is a pure-Python editable install. Optional `flash-attn-3`/`cc_torch` packages for faster inference are described in [the SAM3 README](surgslot-sam3/README.md#installation); they are not required to run SurgSLOT-SAM3.

Verify that the editable installation resolves to this checkout:

```bash
python -c "import sam3; print(sam3.__file__)"
python training/train.py --help
python tools/vos_inference_surgslot.py --help
```

The SAM3 training entry point in this repository is `training/train.py`; its configuration module is rooted at `training/configs/`. Like SAM2, training initializes its semantic text encoder from `openai/clip-vit-base-patch32`; on an offline compute server, download/cache that Hugging Face model before launching training and expose the cache through `HF_HOME` (or the standard Hugging Face cache location). Downloading Meta's official SAM3 checkpoints separately requires requesting access on the [SAM3 Hugging Face repo](https://huggingface.co/facebook/sam3) and authenticating (e.g. `hf auth login`); this is not needed if you only use the released SurgSLOT-SAM3 checkpoint.

## Checkpoints

Large model weights are not committed to Git. Both SurgSLOT checkpoints are hosted on Hugging Face at [`HeverLaw/SurgSLOT`](https://huggingface.co/HeverLaw/SurgSLOT), released under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (non-commercial research use only, no redistribution of modified versions).

| Model | Backbone | Configuration | Checkpoint |
|---|---|---|---|
| SurgSLOT-SAM2 | SAM 2.1 Hiera B+ | [`surgslot_hiera_b+_512.yaml`](surgslot-sam2/sam2/configs/surgslot-sam2/surgslot_hiera_b+_512.yaml) | `checkpoints/surgslot-sam2_hiera_b+.pt` |
| SurgSLOT-SAM3 | SAM 3 | [`surgslot_sam3_sasv.yaml`](surgslot-sam3/training/configs/surgslot-sam3_training/surgslot_sam3_sasv.yaml) | `checkpoints/surgslot-sam3_hiera_b+.pt` |

Download them directly into each backbone's `checkpoints/` directory (filenames already match what the scripts expect, so no renaming is needed):

```bash
mkdir -p surgslot-sam2/checkpoints surgslot-sam3/checkpoints
curl -L -o surgslot-sam2/checkpoints/surgslot-sam2_hiera_b+.pt \
  https://huggingface.co/HeverLaw/SurgSLOT/resolve/main/surgslot-sam2_hiera_b+.pt
curl -L -o surgslot-sam3/checkpoints/surgslot-sam3_hiera_b+.pt \
  https://huggingface.co/HeverLaw/SurgSLOT/resolve/main/surgslot-sam3_hiera_b+.pt
```

Or pass their paths through the environment variables shown below instead of placing them under `checkpoints/`.

Training starts from Meta's official `sam2.1_hiera_base_plus.pt` backbone checkpoint. From `SurgSLOT/surgslot-sam2`, either run `bash checkpoints/download_ckpts.sh` (downloads all four SAM 2.1 checkpoints) or download only the Base+ checkpoint from the [official SAM 2.1 checkpoint directory](https://github.com/facebookresearch/sam2/tree/main/checkpoints). The training configurations expect it at:

```text
surgslot-sam2/checkpoints/sam2.1_hiera_base_plus.pt
```

For SAM3, set `SAM3_CHECKPOINT` to the official SAM3 backbone checkpoint when training, and set `CHECKPOINT_PATH` to the SurgSLOT-SAM3 checkpoint when running inference:

```bash
export SAM3_CHECKPOINT=/path/to/sam3.pt
export CHECKPOINT_PATH=/path/to/surgslot-sam3_hiera_b+.pt
```

## Data Preparation

The benchmark code expects the following layout for each evaluation dataset:

```text
surgslot-sam2/data/<DATASET>/test/
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
cd SurgSLOT/surgslot-sam2
ln -s /path/to/isurg/data data
```

### Quick Demo

To try inference without downloading the full iSurg benchmark, a small demo subset is available — a handful of videos from EndoVis17/EndoVis18 in the same layout as above, under `<DATASET>_demo` instead of `<DATASET>`:

```text
surgslot-sam2/data/VOS-Endovis18_demo/test/{images,vos/Annotations,masks}/...
```

Download the demo data from [`HeverLaw/SurgSLOT-demo`](https://huggingface.co/datasets/HeverLaw/SurgSLOT-demo) on Hugging Face and unzip it under `data/` (the archive contains `VOS-Endovis17_demo/` and `VOS-Endovis18_demo/` at its top level, so unzipping directly into `data/` produces the layout above):

```bash
mkdir -p data
curl -L -o /tmp/VOS-Endovis_demo.zip \
  https://huggingface.co/datasets/HeverLaw/SurgSLOT-demo/resolve/main/VOS-Endovis_demo.zip
unzip /tmp/VOS-Endovis_demo.zip -d data/
```

then place (or symlink) the resulting `data/` under `surgslot-sam2/` and `surgslot-sam3/`. The demo data is released under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) — non-commercial research use only, no redistribution of modified versions. It is a derived subset of EndoVis17/EndoVis18; using it also requires complying with the [original EndoVis data usage terms](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/) ([EndoVis18](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/)).

Both `scripts/test_surgslot.sh` scripts default to `VOS-Endovis18_demo`, so once the data is in place, running the benchmark script with no arguments (besides `CHECKPOINT_PATH`) runs this demo out of the box:

```bash
cd SurgSLOT/surgslot-sam2   # or surgslot-sam3
CHECKPOINT_PATH=/path/to/checkpoint.pt \
bash scripts/test_surgslot.sh
```

Set `DATASETS="VOS-Endovis17_demo VOS-Endovis18_demo"` to run both demo videos, or edit `DATASET_LIST` in the script directly.

## SAM2 Inference

Run from `SurgSLOT/surgslot-sam2`. Long-term memory and all SLM settings default to the paper's configuration (capacity 3, five-frame candidate buffer, confidence threshold 0.9, EMA decay 0.95, six short-term frames), so they do not need to be passed explicitly.

```bash
python tools/vos_inference_surgslot.py \
  --sam2_checkpoint checkpoints/surgslot-sam2_hiera_b+.pt \
  --sam2_cfg configs/surgslot-sam2/surgslot_hiera_b+_512.yaml \
  --base_video_dir ./data/VOS-Endovis17/test/images \
  --input_mask_dir ./data/VOS-Endovis17/test/vos/Annotations \
  --output_mask_dir ./outputs/VOS-Endovis17/point_3 \
  --gt_root ./data/VOS-Endovis17/test/masks \
  --gpu_id 0 \
  --prompt_type point \
  --num_pts 3
```

To run the full benchmark, use the shell script from `SurgSLOT/surgslot-sam2`:

```bash
CHECKPOINT_PATH=/path/to/surgslot-sam2_hiera_b+.pt \
DATA_ROOT=/path/to/isurg/data \
OUTPUT_DIR=./outputs/surgslot-sam2 \
bash scripts/test_surgslot.sh
```

Long-term memory toggles and all SLM thresholds (`--gpu_id`, `--num_long_mem_frame`, `--num_long_mem_candidate`, `--long_mem_confidence_thr`, `--proto_update_thr`, `--ema_alpha_proto`, `--short_term_frame_num`, etc.) already default to the paper's settings inside `tools/vos_inference_surgslot.py`, so the script does not expose them; pass extra flags after the script invocation to override any of them (e.g. `bash scripts/test_surgslot.sh --gpu_id 1`). The script only enables the `point_3` prompt by default; edit the `PROMPT_CONFIGS` array in [`scripts/test_surgslot.sh`](surgslot-sam2/scripts/test_surgslot.sh) to enable the others, and see the script for the remaining environment-variable overrides (dataset selection via `DATASETS`, worker count via `NUM_PROCESSES`, etc.).

## SAM3 Inference

Run from `SurgSLOT/surgslot-sam3`. Long-term memory and all SLM settings default to the paper's configuration and are defined in `sam3/model_builder.py`, so they do not need to be passed explicitly; the tool's `--prompt_type`/`--num_pts` also already default to a three-point prompt.

```bash
python tools/vos_inference_surgslot.py \
  --sam3_checkpoint checkpoints/surgslot-sam3_hiera_b+.pt \
  --base_video_dir ./data/VOS-Endovis17/test/images \
  --input_mask_dir ./data/VOS-Endovis17/test/vos/Annotations \
  --output_mask_dir ./outputs/VOS-Endovis17/point_3 \
  --gt_root ./data/VOS-Endovis17/test/masks \
  --gpu_id 0
```

To run the full benchmark, use the shell script from `SurgSLOT/surgslot-sam3`:

```bash
CHECKPOINT_PATH=/path/to/surgslot-sam3_hiera_b+.pt \
DATA_ROOT=/path/to/isurg/data \
OUTPUT_DIR=./outputs/surgslot-sam3 \
GPU_ID=0 \
bash scripts/test_surgslot.sh
```

Edit the `PROMPT_CONFIGS` array in [`scripts/test_surgslot.sh`](surgslot-sam3/scripts/test_surgslot.sh) to select which of the five prompt settings to run, and see the script for dataset selection via `DATASETS`.

## Training

SurgSLOT training uses the backbone-specific training entry point together with Temporal Semantic Learning losses and long-range memory sampling. Both the SAM2 and SAM3 configurations in this checkout are set to 30 epochs, matching the paper's training recipe. GPU count and per-GPU batch size are currently set for a single-GPU run; scale `--num-gpus`/`--num-nodes` and `scratch.train_batch_size` to reach the paper's total batch size of 20 on 4 GPUs.

Before training:

1. Link or place the training datasets under `SurgSLOT/surgslot-sam2/data` using the paths declared in [`surgslot_sam2_sasv.yaml`](surgslot-sam2/sam2/configs/surgslot-sam2_training/datasets/surgslot_sam2_sasv.yaml).
2. Place Meta's `sam2.1_hiera_base_plus.pt` at `SurgSLOT/surgslot-sam2/checkpoints/sam2.1_hiera_base_plus.pt`.
3. Ensure `openai/clip-vit-base-patch32` is available from Hugging Face or already present in the server cache.

Launch the SAM2 training configuration from `SurgSLOT/surgslot-sam2`:

```bash
cd SurgSLOT/surgslot-sam2
python training/train.py \
  -c configs/surgslot-sam2_training/surgslot_sam2_sasv \
  --use-cluster 0 \
  --num-gpus 1
```

The SAM2 training configuration composes its dataset block from [`surgslot_sam2_sasv.yaml`](surgslot-sam2/sam2/configs/surgslot-sam2_training/datasets/surgslot_sam2_sasv.yaml). Override `--num-gpus` to match the server. Logs and checkpoints are written below `sam2_logs/<config-name>/` unless `launcher.experiment_log_dir` is changed.

Launch the SAM3 training configuration from `SurgSLOT/surgslot-sam3`:

```bash
cd SurgSLOT/surgslot-sam3
export SAM3_CHECKPOINT=/path/to/sam3.pt
python training/train.py \
  -c configs/surgslot-sam3_training/surgslot_sam3_sasv \
  --use-cluster 0 \
  --num-gpus 1
```

The SAM3 configuration uses 672-pixel inputs and 10-frame clips. Logs and checkpoints are written below `sam3_logs/<config-name>/` — mirroring SAM2's `sam2_logs/<config-name>/` layout and kept separate from the `outputs/surgslot-sam3/` benchmark results — unless `SURGSLOT_TRAIN_OUTPUT` is set to redirect them. Use `--num-gpus`/`--num-nodes` to match the server.

## Acknowledgements

This work builds on [SAM 2](https://github.com/facebookresearch/sam2), [SAM 3](https://github.com/facebookresearch/sam3), and [CLIP](https://github.com/openai/CLIP). We thank their authors for making these resources available.

iSurg consolidates data from multiple surgical datasets. We thank the dataset creators and National University Hospital for supporting evaluation on in-house hysterectomy videos. Users must request or download datasets from their respective official sources.

## License

This is a **multi-license repository**:

- Original SurgSLOT code and documentation are released under the [Apache License 2.0](LICENSE), unless a more specific license applies in a subdirectory.
- The `surgslot-sam2/` directory is based on Meta's SAM 2 and remains subject to its [Apache License 2.0](surgslot-sam2/LICENSE). Additional third-party terms are retained in the [cc_torch license](surgslot-sam2/LICENSE_cctorch), [SA-V dataset license](surgslot-sam2/sav_dataset/LICENSE), [DAVIS license](surgslot-sam2/sav_dataset/LICENSE_DAVIS), and [VOS benchmark license](surgslot-sam2/sav_dataset/LICENSE_VOS_BENCHMARK).
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
