# Temporal-stage J&F evaluation
# Splits each video into 5 temporal stages (0-20%, 20-40%, ..., 80-100%)
# and computes J&F independently for each stage.

import math
import os
import time
import json
from collections import defaultdict
from multiprocessing import Pool
from os import path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image
from skimage.morphology import disk

# ── Reused metric helpers ──────────────────────────────────────────────

def _seg2bmap(seg, width=None, height=None):
    seg = seg.astype(bool)
    seg[seg > 0] = 1
    assert np.atleast_3d(seg).shape[2] == 1
    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height
    h, w = seg.shape[:2]
    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)
    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can't convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1
    return bmap


def get_iou(intersection, pixel_sum):
    if intersection == pixel_sum:
        assert intersection == 0
        return 1
    return intersection / (pixel_sum - intersection)


def compute_boundary_f(obj_mask, obj_gt, boundary_disk):
    mask_boundary = _seg2bmap(obj_mask)
    gt_boundary = _seg2bmap(obj_gt)
    mask_dilated = cv2.dilate(mask_boundary.astype(np.uint8), boundary_disk)
    gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), boundary_disk)

    gt_match = gt_boundary * mask_dilated
    fg_match = mask_boundary * gt_dilated
    n_fg = np.sum(mask_boundary)
    n_gt = np.sum(gt_boundary)

    if n_fg == 0 and n_gt > 0:
        precision, recall = 1, 0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0, 1
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1, 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Per-frame metric collector (no aggregation yet) ────────────────────

def compute_frame_metrics(obj_mask, obj_gt, boundary_disk):
    """Return (iou, boundary_f) for a single frame & object."""
    iou = get_iou((obj_mask * obj_gt).sum(), obj_mask.sum() + obj_gt.sum())
    bf = compute_boundary_f(obj_mask, obj_gt, boundary_disk)
    return iou, bf


# ── Temporal-stage video evaluator ─────────────────────────────────────

STAGE_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
STAGE_NAMES = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


class TemporalVideoEvaluator:
    def __init__(self, gt_root, pred_root, skip_first_and_last=True, boundary=0.008):
        self.gt_root = gt_root
        self.pred_root = pred_root
        self.skip_first_and_last = skip_first_and_last
        self.boundary = boundary

    def __call__(self, data):
        vid_name, evaluate_object_ids = data
        to_evaluate, is_sav_format = self._scan_vid_folder(vid_name, evaluate_object_ids)

        # result per object: obj_id -> {stage_name: {"J": [...], "F": [...]}}
        obj_stage_metrics = {}

        for all_frames, obj_dir, gt_path, pred_path in to_evaluate:
            obj_id = int(obj_dir)
            frames = list(all_frames)  # copy
            if self.skip_first_and_last:
                frames = frames[1:-1]

            n_frames = len(frames)
            if n_frames == 0:
                continue

            # Collect per-frame J and F
            frame_j_list = []
            frame_f_list = []
            for frame in frames:
                gt_array, pred_array = self._get_gt_and_pred(
                    gt_path, pred_path, frame, is_sav_format, object_id=obj_id
                )
                bound_pix = np.ceil(self.boundary * np.linalg.norm(gt_array.shape))
                boundary_disk = disk(bound_pix)

                obj_mask = pred_array if not is_sav_format else pred_array
                obj_gt = gt_array if not is_sav_format else gt_array
                # already binary after _get_gt_and_pred when is_sav_format

                gt_objects = np.unique(obj_gt)
                gt_objects = gt_objects[gt_objects != 0].tolist()
                mask_objects = np.unique(obj_mask)
                mask_objects = mask_objects[mask_objects != 0].tolist()
                all_objects = set(gt_objects) | set(mask_objects)

                # For SA-V format (binary masks), the only object index is True/1
                if is_sav_format:
                    all_objects = {True} if len(all_objects) > 0 else set()
                    if len(all_objects) == 0:
                        # both empty
                        frame_j_list.append(1.0)
                        frame_f_list.append(1.0)
                        continue
                    om = obj_mask.astype(bool)
                    og = obj_gt.astype(bool)
                    j_val, f_val = compute_frame_metrics(om, og, boundary_disk)
                    frame_j_list.append(j_val)
                    frame_f_list.append(f_val)
                else:
                    # multi-object in single mask (DAVIS style)
                    j_vals, f_vals = [], []
                    for oi in all_objects:
                        om = obj_mask == oi
                        og = obj_gt == oi
                        j_val, f_val = compute_frame_metrics(om, og, boundary_disk)
                        j_vals.append(j_val)
                        f_vals.append(f_val)
                    frame_j_list.append(np.mean(j_vals) if j_vals else 1.0)
                    frame_f_list.append(np.mean(f_vals) if f_vals else 1.0)

            # ── Split into temporal stages ──
            stage_metrics = {}
            for stage_name, (lo, hi) in zip(STAGE_NAMES, STAGE_BINS):
                idx_start = int(round(lo * n_frames))
                idx_end = int(round(hi * n_frames))
                if idx_start == idx_end:
                    # edge case: very short video, ensure at least 1 frame per bin
                    idx_end = min(idx_start + 1, n_frames)
                j_slice = frame_j_list[idx_start:idx_end]
                f_slice = frame_f_list[idx_start:idx_end]
                if len(j_slice) > 0:
                    mean_j = np.mean(j_slice) * 100
                    mean_f = np.mean(f_slice) * 100
                else:
                    mean_j = float("nan")
                    mean_f = float("nan")
                stage_metrics[stage_name] = {
                    "J": mean_j,
                    "F": mean_f,
                    "J&F": (mean_j + mean_f) / 2,
                    "n_frames": len(j_slice),
                }

            # Also store the overall (all frames)
            overall_j = np.mean(frame_j_list) * 100
            overall_f = np.mean(frame_f_list) * 100
            stage_metrics["overall"] = {
                "J": overall_j,
                "F": overall_f,
                "J&F": (overall_j + overall_f) / 2,
                "n_frames": n_frames,
            }

            obj_stage_metrics[obj_id] = stage_metrics

        return vid_name, obj_stage_metrics

    # ── helpers (same logic as original) ──

    def _get_gt_and_pred(self, gt_path, pred_path, f_name, is_sav_format, object_id=0):
        gt_mask_path = path.join(gt_path, f_name)
        pred_mask_path = path.join(pred_path, f_name)
        assert os.path.exists(pred_mask_path), f"{pred_mask_path} not found"

        gt_array = np.array(Image.open(gt_mask_path))
        pred_array = np.array(Image.open(pred_mask_path))
        assert gt_array.shape[-2:] == pred_array.shape[-2:]

        if is_sav_format:
            gt_array = (gt_array == object_id).astype(np.uint8)
            pred_array = (pred_array == object_id).astype(np.uint8)

        return gt_array, pred_array

    def _scan_vid_folder(self, vid_name, evaluate_object_ids):
        vid_gt_path = path.join(self.gt_root, vid_name)
        vid_pred_path = path.join(self.pred_root, vid_name)
        all_files_and_dirs = sorted(os.listdir(vid_pred_path))
        to_evaluate = []
        is_sav_format = True
        for obj_dir in all_files_and_dirs:
            if (
                not obj_dir.isdigit()
                or not path.isdir(path.join(vid_pred_path, obj_dir))
                or int(obj_dir) not in evaluate_object_ids
            ):
                continue
            obj_gt_path = vid_gt_path
            obj_pred_path = path.join(vid_pred_path, obj_dir)
            frames = sorted(os.listdir(obj_pred_path))
            frames = [f for f in frames if os.path.exists(path.join(obj_gt_path, f))]
            to_evaluate.append((frames, obj_dir, obj_gt_path, obj_pred_path))
        return to_evaluate, is_sav_format


# ── Main benchmark entry point ─────────────────────────────────────────

def temporal_benchmark(
    gt_roots,
    mask_roots,
    strict=True,
    num_processes=None,
    verbose=True,
    skip_first_and_last=True,
    evaluate_object_id_list=None,
    output_json=None,
):
    """
    Same interface as the original benchmark(), but reports J&F per temporal stage.
    """
    assert len(gt_roots) == len(mask_roots)
    if evaluate_object_id_list is None:
        evaluate_object_id_list = range(1, 256)

    pool = Pool(num_processes)

    for gt_root, mask_root in zip(gt_roots, mask_roots):
        gt_videos = sorted(
            [v for v in os.listdir(gt_root) if path.isdir(path.join(gt_root, v))]
        )
        mask_videos = sorted(
            [v for v in os.listdir(mask_root) if path.isdir(path.join(mask_root, v))]
        )

        if strict:
            videos = sorted(gt_videos)
        else:
            videos = sorted(list(set(gt_videos) & set(mask_videos)))

        print(f"Evaluating {len(videos)} videos from {gt_root}")

        input_data = [(v, evaluate_object_id_list) for v in videos]
        evaluator = TemporalVideoEvaluator(
            gt_root, mask_root, skip_first_and_last=skip_first_and_last
        )
        results = list(
            tqdm.tqdm(pool.imap(evaluator, input_data), total=len(videos))
        )

    pool.close()

    # ── Aggregate across all videos & objects ──
    # stage_name -> {"J": [...], "F": [...]}
    global_stage = {s: {"J": [], "F": []} for s in STAGE_NAMES + ["overall"]}
    per_video_detail = {}  # vid -> obj -> stage -> metrics

    for vid_name, obj_stage_metrics in results:
        per_video_detail[vid_name] = {}
        for obj_id, stage_dict in obj_stage_metrics.items():
            per_video_detail[vid_name][obj_id] = stage_dict
            for stage_name, m in stage_dict.items():
                if not np.isnan(m["J"]):
                    global_stage[stage_name]["J"].append(m["J"])
                    global_stage[stage_name]["F"].append(m["F"])

    # ── Print report ──
    header = f"{'Stage':<12} {'J&F':>7} {'J':>7} {'F':>7} {'#obj-frames':>12}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("  Global Temporal-Stage Results")
    print(sep)
    print(header)
    print(sep)

    summary = {}
    for stage_name in STAGE_NAMES + ["overall"]:
        j_arr = np.array(global_stage[stage_name]["J"])
        f_arr = np.array(global_stage[stage_name]["F"])
        if len(j_arr) == 0:
            continue
        mean_j = j_arr.mean()
        mean_f = f_arr.mean()
        mean_jf = (mean_j + mean_f) / 2
        n = len(j_arr)
        tag = stage_name if stage_name != "overall" else "** overall **"
        print(f"{tag:<12} {mean_jf:>7.2f} {mean_j:>7.2f} {mean_f:>7.2f} {n:>12d}")
        summary[stage_name] = {"J&F": mean_jf, "J": mean_j, "F": mean_f, "count": n}

    print(sep)

    # ── Optionally print per-video detail ──
    if verbose:
        print("\nPer-video / per-object breakdown:")
        for vid_name in sorted(per_video_detail):
            for obj_id in sorted(per_video_detail[vid_name]):
                sd = per_video_detail[vid_name][obj_id]
                stages_str = "  ".join(
                    f"{s}: J&F={sd[s]['J&F']:.1f}" for s in STAGE_NAMES if s in sd
                )
                print(f"  {vid_name} obj={obj_id:03d}  {stages_str}")

    # ── Save JSON ──
    out_path = output_json or path.join(mask_roots[0], "temporal_results.json")
    save_data = {"global": summary, "per_video": {}}
    for vid_name, obj_dict in per_video_detail.items():
        save_data["per_video"][vid_name] = {
            str(k): v for k, v in obj_dict.items()
        }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary, per_video_detail


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Temporal-stage J&F evaluation")
    parser.add_argument("--gt_root", required=True, help="Path to GT masks")
    parser.add_argument("--pred_root", required=True, help="Path to predicted masks")
    parser.add_argument("--num_processes", type=int, default=None)
    parser.add_argument("--skip_first_and_last", action="store_true", default=True)
    parser.add_argument("--no_skip_first_and_last", dest="skip_first_and_last", action="store_false")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    temporal_benchmark(
        gt_roots=[args.gt_root],
        mask_roots=[args.pred_root],
        num_processes=args.num_processes,
        skip_first_and_last=args.skip_first_and_last,
        output_json=args.output_json,
    )