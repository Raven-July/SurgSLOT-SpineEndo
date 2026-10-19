#!/usr/bin/env python3
"""Sparse LabelMe prompts -> SurgSLOT-SAM3 bidirectional assisted annotation.

Refine human regions by default, then propagate the accepted anchor masks.
Original shapes are retained as provenance; rejected candidates fall back to them.
Missing labels are never interpreted as negative labels or background.
"""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SAM3_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT.parent / "Data/邢静远Segment0006_t193p000s_fps10_n90"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def shape_mask(shape, height, width):
    """Match LabelMe/PIL rasterization, including the circle in the actual data."""
    points = [tuple(p) for p in shape["points"]]
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    kind = shape.get("shape_type", "polygon")
    if kind == "polygon" and len(points) >= 3:
        draw.polygon(points, fill=1)
    elif kind == "circle" and len(points) == 2:
        (x, y), (x2, y2) = points
        radius = float(np.hypot(x2 - x, y2 - y))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
    elif kind == "rectangle" and len(points) == 2:
        (x, y), (x2, y2) = points
        draw.rectangle((min(x, x2), min(y, y2), max(x, x2), max(y, y2)), fill=1)
    else:
        raise ValueError(
            f"Unsupported/invalid LabelMe region: {kind}, {len(points)} points"
        )
    return np.asarray(canvas, dtype=np.uint8).copy()


def filter_source_shapes(document, height, width, args):
    """Record tool speckles without altering source geometry or its indices."""
    info = []
    for index, shape in enumerate(document.get("shapes", [])):
        mask = shape_mask(shape, height, width)
        ys, xs = np.nonzero(mask)
        info.append(
            (
                index,
                shape,
                int(mask.sum()),
                int(np.ptp(xs)) + 1 if len(xs) else 0,
                int(np.ptp(ys)) + 1 if len(ys) else 0,
            )
        )
    largest = {}
    for _, shape, area, _, _ in info:
        largest[shape["label"]] = max(largest.get(shape["label"], 0), area)
    removed = []
    for index, shape, area, width_, height_ in info:
        flags = shape.get("flags") or {}
        if area and (
            flags.get("keep_small") or flags.get("preserve") or flags.get("no_refine")
        ):
            continue
        protected = shape["label"] in args.protect_small_class
        tiny = area < args.source_min_area and not protected
        speckle = (
            not protected
            and area <= args.artifact_max_area
            and max(width_, height_) <= args.artifact_max_size
            and largest[shape["label"]] >= 4096
            and area <= largest[shape["label"]] * 0.02
        )
        if area == 0 or tiny or speckle:
            removed.append(
                {
                    "shape_index": index,
                    "label": shape["label"],
                    "area": area,
                    "size": [width_, height_],
                    "reason": "empty" if not area else "tool_speckle",
                }
            )
    if removed:
        document["surgslot_filtered_shapes"] = removed
    return removed


def valid_source_shapes(document):
    removed = {r["shape_index"] for r in document.get("surgslot_filtered_shapes", [])}
    return [
        (i, shape)
        for i, shape in enumerate(document.get("shapes", []))
        if i not in removed
    ]


def read_sequence(input_dir, anchor_every=1, args=None):
    frames = sorted(
        (p for p in input_dir.iterdir() if p.suffix.lower() in EXTENSIONS),
        key=natural_key,
    )
    if not frames or len({p.stem for p in frames}) != len(frames):
        raise ValueError("No images, or duplicate image stems")
    documents, sizes, hashes, anchors = [], [], {}, {}
    for index, frame in enumerate(frames):
        with Image.open(frame) as image:
            width, height = image.size
        sizes.append((height, width))
        ann, doc = frame.with_suffix(".json"), None
        if ann.exists():
            raw = ann.read_bytes()
            hashes[ann.name] = hashlib.sha256(raw).hexdigest()
            original = json.loads(raw)
            # Explicitly held-out JSONs are evaluation data, never output hints.
            if index % anchor_every == 0:
                doc = original
                if (doc.get("imageHeight"), doc.get("imageWidth")) != (height, width):
                    raise ValueError(f"Image/JSON dimensions differ: {ann}")
                if args is not None:
                    filter_source_shapes(doc, height, width, args)
                for _, shape in valid_source_shapes(doc):
                    if not shape_mask(shape, height, width).any():
                        raise ValueError(
                            f"Empty/out-of-frame region in {ann}: {shape['label']}"
                        )
                    anchors.setdefault(shape["label"], set()).add(index)
        documents.append(doc)
    if len(set(sizes)) != 1 or not anchors:
        raise ValueError(
            "All frames must have equal dimensions and at least one annotated region"
        )
    return (
        frames,
        documents,
        {k: sorted(v) for k, v in sorted(anchors.items())},
        sizes[0],
        hashes,
    )


def selected_for_refinement(shape, index, args):
    flags = shape.get("flags") or {}
    return (
        args.refine
        and not (flags.get("preserve") or flags.get("no_refine"))
        and (not args.refine_label or shape["label"] in args.refine_label)
        and (not args.refine_frame or index + 1 in args.refine_frame)
    )


def make_tasks(anchors, documents, args):
    """Each missing frame receives only its adjacent same-class anchors."""
    tasks, nframes = [], len(documents)
    for label, indices in anchors.items():
        for pos, anchor in enumerate(indices):
            previous = indices[pos - 1] if pos else -1
            following = indices[pos + 1] if pos + 1 < len(indices) else nframes
            spans = []
            if anchor - previous > 1:
                spans.append({"reverse": True, "end": previous + 1})
            if following - anchor > 1:
                spans.append({"reverse": False, "end": following - 1})
            refine = [
                i
                for i, s in valid_source_shapes(documents[anchor])
                if s["label"] == label and selected_for_refinement(s, anchor, args)
            ]
            if spans or refine:
                tasks.append(
                    {
                        "id": len(tasks),
                        "label": label,
                        "anchor": anchor,
                        "spans": spans,
                        "refine": refine,
                    }
                )
    if args.refine and args.temporal_stride > 0:
        lookup = {(t["label"], t["anchor"]): t for t in tasks}
        for label, indices in anchors.items():
            if len(indices) < 2 or (
                args.refine_label and label not in args.refine_label
            ):
                continue
            keys = [indices[0]]
            for index in indices[1:]:
                if index - keys[-1] >= args.temporal_stride:
                    keys.append(index)
            if keys[-1] != indices[-1]:
                keys.append(indices[-1])
            for position, anchor in enumerate(keys):
                task = lookup.get((label, anchor))
                if task is None:
                    task = {
                        "id": len(tasks),
                        "label": label,
                        "anchor": anchor,
                        "refine": [],
                        "spans": [],
                    }
                    tasks.append(task)
                spans = []
                if position:
                    spans.append({"reverse": True, "end": keys[position - 1]})
                if position + 1 < len(keys):
                    spans.append({"reverse": False, "end": keys[position + 1]})
                task["temporal_spans"] = spans
    return tasks


def components(mask):
    count, ids = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    for index in range(1, count):
        yield (ids == index).astype(np.uint8)


def positive_core(mask, radius=6):
    if not mask.any():
        return mask.copy()
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    radius = min(radius, max(1.0, float(distance.max()) * 0.4))
    return (distance > radius).astype(np.uint8)


def refinement_points(mask, padding, num_points=3, dilate=5):
    """Expanded box and distributed interior positives, no exterior negatives."""
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    radius = min(dilate, max(1, int(np.sqrt(len(xs)) * 0.03)))
    px = max(2, (x1 - x0) * padding) + radius
    py = max(2, (y1 - y0) * padding) + radius
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    interior = distance[ys, xs] >= max(1, float(distance.max()) * 0.3)
    candidates = np.column_stack((xs[interior], ys[interior]))
    candidates = candidates[:: max(1, len(candidates) // 4096)]
    depth = distance[candidates[:, 1], candidates[:, 0]]
    selected = [candidates[int(depth.argmax())]]
    nearest = np.full(len(candidates), np.inf)
    for _ in range(min(num_points, len(candidates)) - 1):
        nearest = np.minimum(nearest, ((candidates - selected[-1]) ** 2).sum(axis=1))
        selected.append(candidates[int((nearest * depth).argmax())])
    points = np.array(
        [
            [max(0, x0 - px), max(0, y0 - py)],
            [min(width - 1, x1 + px), min(height - 1, y1 + py)],
            *selected,
        ],
        dtype=np.float32,
    ) / [width, height]
    return points.astype(np.float32), np.array(
        [2, 3] + [1] * len(selected), dtype=np.int32
    )


def check_refinement(seed, prediction, args, rough=False):
    """Reject displacement/leakage; allow editing in a bounded neighbourhood."""
    seed = seed.astype(np.uint8)
    prediction = prediction.astype(np.uint8)
    area = int(seed.sum())
    count, ids = cv2.connectedComponents(prediction, connectivity=8)
    touched = np.unique(ids[seed.astype(bool)])
    touched = touched[touched != 0]
    candidate = (
        np.isin(ids, touched).astype(np.uint8) if len(touched) else np.zeros_like(seed)
    )
    ratio = float(candidate.sum() / area)
    metrics = {
        "seed_coverage": float((candidate & seed).sum() / area),
        "area_ratio": ratio,
        "clipped_pixels": 0,
    }
    if not prediction.any():
        return seed.copy(), False, "empty", metrics
    if not candidate.any():
        return seed.copy(), False, "displaced", metrics
    if ratio > args.max_area_ratio:
        return seed.copy(), False, "excessive_area", metrics
    ys, xs = np.nonzero(seed)
    radius = min(
        args.max_expand_px, max(2.0, 0.5 * min(np.ptp(xs) + 1, np.ptp(ys) + 1))
    )
    outside_distance = cv2.distanceTransform(1 - seed, cv2.DIST_L2, 5)
    bounded = candidate & (outside_distance <= radius).astype(np.uint8)
    metrics["clipped_pixels"] = int(candidate.sum() - bounded.sum())
    metrics["seed_coverage"] = float((bounded & seed).sum() / area)
    metrics["area_ratio"] = float(bounded.sum() / area)
    minimum = args.rough_seed_coverage if rough else args.min_seed_coverage
    if metrics["seed_coverage"] < minimum:
        return seed.copy(), False, "lost_seed", metrics
    core = positive_core(seed, args.core_radius) if not rough else np.zeros_like(seed)
    metrics["restored_core_pixels"] = int(np.count_nonzero(core & (1 - bounded)))
    bounded |= core
    metrics["seed_coverage"] = float((bounded & seed).sum() / area)
    metrics["area_ratio"] = float(bounded.sum() / area)
    return bounded, True, "accepted", metrics


def refine_shape(predictor, state, mask, shape, args):
    """Decode in an isolated single-frame state; retry uncovered positive cores."""
    refined, trials, accepted_regions = np.zeros_like(mask), [], 0
    rough = bool((shape.get("flags") or {}).get("rough"))
    for region in components(mask):
        points, labels = refinement_points(
            region, args.box_padding, args.refine_points, args.prompt_dilate
        )
        for attempt in range(args.refine_attempts):
            predictor.clear_all_points_in_video(state)
            _, _, _, logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
                clear_old_points=True,
            )
            prediction = (
                (logits[0, 0].sigmoid() > args.mask_threshold)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            candidate, accepted, reason, metrics = check_refinement(
                region, prediction, args, rough
            )
            trials.append({"attempt": attempt + 1, "reason": reason, **metrics})
            if accepted:
                accepted_regions += 1
                break
            missing = region & (1 - prediction)
            if not missing.any():
                break
            distance = cv2.distanceTransform(missing, cv2.DIST_L2, 5)
            y, x = np.unravel_index(distance.argmax(), distance.shape)
            points = np.concatenate(
                [points, [[x / mask.shape[1], y / mask.shape[0]]]]
            ).astype(np.float32)
            labels = np.append(labels, np.int32(1))
        refined |= candidate
    return refined, {
        "accepted": accepted_regions > 0,
        "accepted_regions": accepted_regions,
        "trials": trials,
        "changed_pixels": int(np.count_nonzero(refined != mask)),
        "seed_coverage": float((refined & mask).sum() / mask.sum()),
        "area_ratio": float(refined.sum() / mask.sum()),
        "iou_with_original": float(
            (refined & mask).sum() / max(1, (refined | mask).sum())
        ),
    }


def prediction_path(work, task, reverse, index, stream="prediction"):
    return Path(work) / f"{stream}_{task['id']:05d}_{int(reverse)}_{index:05d}.npz"


class ReversedFrames:
    """Index view over CPU frames; does not duplicate the image tensor."""

    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.frames[len(self) - 1 - index]


def propagate_region(predictor, state, region, anchor, end, reverse):
    import torch

    # Native reverse=True works for short memory, but the current SurgSLOT
    # _update_long_mem_candidates uses frame_idx - 1. Reversing the frame view
    # lets the unchanged forward path update long memory in both directions.
    images = state["images"]
    nframes = state["num_frames"]
    predictor.clear_all_points_in_video(state)
    state["cached_features"].clear()  # Same index must not refer to another image.
    if reverse:
        state["images"] = ReversedFrames(images)
    model_anchor = nframes - 1 - anchor if reverse else anchor
    try:
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=model_anchor,
            obj_id=1,
            mask=torch.from_numpy(region).float(),
        )
        for index, _, _, logits, presence in predictor.propagate_in_video(
            inference_state=state,
            start_frame_idx=model_anchor,
            max_frame_num_to_track=abs(end - anchor),
            reverse=False,
            propagate_preflight=True,
            tqdm_disable=True,
        ):
            index = nframes - 1 - index if reverse else index
            yield index, logits, presence
    finally:
        predictor.clear_all_points_in_video(state)
        state["images"] = images
        state["cached_features"].clear()


def run_task(predictor, state, task, documents, work, args):
    import torch

    anchor, label = task["anchor"], task["label"]
    height, width = state["video_height"], state["video_width"]
    seed, refinements = np.zeros((height, width), dtype=np.uint8), []
    refine_state = None
    if task["refine"]:
        # No propagation memory, reversed frame view, or frame-index cache can
        # affect anchor refinement. Reuse only this actual frame's image features.
        cached = state["cached_features"].get(anchor)
        refine_state = predictor.init_state(
            video_height=height,
            video_width=width,
            num_frames=1,
            cached_features={0: cached} if cached is not None else None,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        refine_state["images"] = [state["images"][anchor]]
    for shape_index, shape in valid_source_shapes(documents[anchor]):
        if shape["label"] != label:
            continue
        mask = shape_mask(shape, height, width)
        if shape_index in task["refine"]:
            mask, record = refine_shape(predictor, refine_state, mask, shape, args)
            record.update(
                {
                    "frame": anchor,
                    "frame_1based": anchor + 1,
                    "shape_index": shape_index,
                    "label": label,
                }
            )
            refinements.append(record)
            np.savez_compressed(
                Path(work) / f"refined_{anchor:05d}_{shape_index:05d}.npz", mask=mask
            )
        seed |= mask
    if refine_state is not None:
        cached = refine_state["cached_features"].get(0)
        if cached is not None:
            state["cached_features"] = {anchor: cached}
        predictor.clear_all_points_in_video(refine_state)
        del refine_state
    region_count = 0
    # Independent temporary components avoid both mutual exclusion and large
    # object batches. add_new_mask itself edits other overlapping mask prompts.
    for region in components(seed):
        region_count += 1
        propagation_spans = [(s, "prediction") for s in task["spans"]]
        propagation_spans += [(s, "temporal") for s in task.get("temporal_spans", [])]
        for span, stream in propagation_spans:
            for index, logits, presence in propagate_region(
                predictor, state, region, anchor, span["end"], span["reverse"]
            ):
                if index == anchor:
                    continue
                probability = logits[0, 0].float().sigmoid()
                if float(presence.reshape(-1)[0].sigmoid()) < args.presence_threshold:
                    probability.zero_()
                probability = (probability * 255).round().to(torch.uint8).cpu().numpy()
                path = prediction_path(work, task, span["reverse"], index, stream)
                if path.exists():
                    with np.load(path) as old:
                        probability = np.maximum(probability, old["probability"])
                np.savez_compressed(path, probability=probability)
        predictor.clear_all_points_in_video(state)
    return {
        "task": task["id"],
        "label": label,
        "anchor": anchor,
        "components": region_count,
        "refinements": refinements,
    }


def gpu_worker(gpu, tasks, documents, frame_dir, work, args):
    """One unchanged SurgSLOT tracker per GPU; fresh memory per direction/region."""
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("USE_PERFLIB", "0")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        sys.path.insert(0, str(SAM3_ROOT))
        import torch

        torch.set_num_threads(2)
        torch.cuda.set_device(gpu)
        from sam3.model_builder import build_surgslot_sam3_predictor

        print(
            f"[GPU {gpu}] loading SurgSLOT-SAM3; {len(tasks)} anchor tasks", flush=True
        )
        predictor = (
            build_surgslot_sam3_predictor(
                checkpoint_path=str(args.checkpoint),
                image_size=args.image_size,
                use_fa3=False,
                apply_postprocessing=False,
            )
            .cuda()
            .eval()
        )
        predictor.non_overlap_masks_for_output = False
        predictor.non_overlap_masks_for_mem_enc = False
        state = predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
        results = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for task in tasks:
                started = time.monotonic()
                result = run_task(predictor, state, task, documents, work, args)
                results.append(result)
                print(
                    f"[GPU {gpu}] {task['label']} anchor={task['anchor'] + 1}: "
                    f"{result['components']} components, {time.monotonic() - started:.1f}s",
                    flush=True,
                )
        torch.cuda.synchronize()
        write_json(
            Path(work) / f"worker_{gpu}.json",
            {
                "gpu": gpu,
                "device": torch.cuda.get_device_name(gpu),
                "pid": os.getpid(),
                "peak_allocated_mb": round(
                    torch.cuda.max_memory_allocated() / 2**20, 1
                ),
                "results": results,
            },
        )
    except Exception:
        write_json(
            Path(work) / f"worker_{gpu}_error.json",
            {"traceback": traceback.format_exc()},
        )
        traceback.print_exc()
        raise


def launch_workers(gpus, tasks, documents, frame_dir, work, args):
    buckets, loads = [[] for _ in gpus], [0 for _ in gpus]
    groups = {}
    for task in tasks:
        groups.setdefault(task["anchor"], []).append(task)

    def cost(group):
        return sum(
            2
            + len(t["refine"])
            + sum(abs(s["end"] - t["anchor"]) for s in t["spans"])
            + sum(abs(s["end"] - t["anchor"]) for s in t.get("temporal_spans", []))
            for t in group
        )

    for group in sorted(groups.values(), key=cost, reverse=True):
        target = min(range(len(gpus)), key=lambda i: loads[i])
        buckets[target].extend(group)
        loads[target] += cost(group)
    context = mp.get_context("spawn")
    workers = [
        context.Process(
            target=gpu_worker, args=(gpu, bucket, documents, frame_dir, work, args)
        )
        for gpu, bucket in zip(gpus, buckets)
    ]
    try:
        for worker in workers:
            worker.start()
        while any(worker.is_alive() for worker in workers):
            for worker in workers:
                worker.join(timeout=0.2)
            if any(worker.exitcode not in (None, 0) for worker in workers):
                raise RuntimeError(f"GPU worker failed; see {work}/worker_*_error.json")
        if any(worker.exitcode != 0 for worker in workers):
            raise RuntimeError(f"GPU worker failed; see {work}")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            if worker.pid is not None:
                worker.join()
    return [json.loads((Path(work) / f"worker_{gpu}.json").read_text()) for gpu in gpus]


def fuse_predictions(candidates, threshold=0.5):
    """Distance-weighted probability fusion, including empty predictions."""
    weights = np.array([1.0 / max(1, distance) for _, distance in candidates])
    probability = sum(
        p.astype(np.float32) * float(w) / 255 for (p, _), w in zip(candidates, weights)
    ) / float(weights.sum())
    iou = None
    if len(candidates) == 2:
        a, b = (p.astype(np.float32) / 255 > threshold for p, _ in candidates)
        union = np.count_nonzero(a | b)
        iou = float(np.count_nonzero(a & b) / union) if union else 1.0
    return (probability > threshold).astype(np.uint8), iou


def mask_shapes(mask, label, min_area=8, epsilon=1.0, flags=None, description=""):
    """Split hole-bearing masks into ordinary polygons, preserving their holes.

    Adjacent same-label polygons meet at a cut; their union has no filled hole.
    This avoids undoing class-boundary coordination during LabelMe conversion.
    """
    shapes, holes = [], 0

    def collect(part, xoffset=0, yoffset=0):
        nonlocal holes
        if not part.any():
            return
        contours, hierarchy = cv2.findContours(
            part, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        inner = [i for i in range(len(contours)) if hierarchy[0, i, 3] != -1]
        if inner:
            contour = contours[max(inner, key=lambda i: cv2.contourArea(contours[i]))]
            x, y, width, height = cv2.boundingRect(contour)
            holes += 1
            # A line through the hole opens it to each sub-mask's exterior.
            if part.shape[1] >= part.shape[0] and part.shape[1] > 2:
                cut = min(part.shape[1] - 1, max(1, x + width // 2))
                collect(part[:, :cut].copy(), xoffset, yoffset)
                collect(part[:, cut:].copy(), xoffset + cut, yoffset)
            else:
                cut = min(part.shape[0] - 1, max(1, y + height // 2))
                collect(part[:cut].copy(), xoffset, yoffset)
                collect(part[cut:].copy(), xoffset, yoffset + cut)
            return
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(polygon) < 3:
                continue
            polygon = polygon + [xoffset, yoffset]
            shapes.append(
                {
                    "label": label,
                    "points": polygon.astype(float).tolist(),
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": dict(flags or {}),
                    "description": description,
                    "mask": None,
                }
            )

    collect(mask.astype(np.uint8))
    return shapes, holes


def class_masks(shapes, height, width):
    masks = {}
    for shape in shapes:
        masks.setdefault(shape["label"], np.zeros((height, width), dtype=np.uint8))
        masks[shape["label"]] |= shape_mask(shape, height, width)
    return masks


def overlap_pairs(masks, automatic_labels, original_masks=None):
    conflicts, manual = [], []
    labels = sorted(masks)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            overlap = masks[left] & masks[right]
            if (
                original_masks is not None
                and left in original_masks
                and right in original_masks
            ):
                overlap = overlap & (1 - (original_masks[left] & original_masks[right]))
            pixels = int(np.count_nonzero(overlap))
            if pixels:
                item = {"labels": [left, right], "pixels": pixels}
                (conflicts if {left, right} & automatic_labels else manual).append(item)
    return conflicts, manual


def apply_refinements(document, frame_index, records, work, args):
    """Replace accepted source shapes in the actual exported candidate document."""
    doc = copy.deepcopy(document)
    selected = {(r["frame"], r["shape_index"]): r for r in records}
    shapes = []
    for index, shape in valid_source_shapes(doc):
        record = selected.get((frame_index, index))
        if record is None or not record["accepted"] or not record["changed_pixels"]:
            shapes.append(shape)
            if record is not None:
                record["export_changed"] = False
            continue
        with np.load(Path(work) / f"refined_{frame_index:05d}_{index:05d}.npz") as data:
            polygons, holes = mask_shapes(
                data["mask"], shape["label"], args.min_area, args.polygon_epsilon
            )
        record["holes_split"] = holes
        record["export_changed"] = bool(polygons)
        record["exported_polygons"] = len(polygons)
        if not polygons:
            record["export_reason"] = "contour_filtered_fallback"
            shapes.append(shape)
            continue
        for polygon in polygons:
            flags = dict(shape.get("flags") or {})
            flags.update(
                automatic=True,
                refined=True,
                needs_review=bool(
                    holes or any(t["clipped_pixels"] for t in record["trials"])
                ),
            )
            polygon.update(
                group_id=shape.get("group_id"),
                flags=flags,
                description=shape.get("description", ""),
                source_shape_index=index,
            )
        shapes.extend(polygons)
    doc["shapes"] = shapes
    if any(r["frame"] == frame_index for r in records) or doc.get(
        "surgslot_filtered_shapes"
    ):
        doc["surgslot_original_shapes"] = copy.deepcopy(document["shapes"])
        doc["surgslot_refinement"] = [r for r in records if r["frame"] == frame_index]
    return doc


def clean_candidate(mask, seed, label, args):
    result = mask.astype(np.uint8).copy()
    if label not in args.protect_small_class and args.boundary_smooth:
        radius = args.boundary_smooth
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        result = cv2.medianBlur(result, 3)
    n, ids, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    minimum = (
        args.min_area if label in args.protect_small_class else args.min_component_area
    )
    for component in range(1, n):
        if stats[component, cv2.CC_STAT_AREA] < minimum and not np.any(
            seed[ids == component]
        ):
            result[ids == component] = 0
    return result


def temporal_consensus(local, seed, candidates, args, trusted_seed=None):
    weights = np.array([1.0 / max(1, distance) for _, distance in candidates])
    probability = sum(
        p.astype(np.float32) / 255 * float(w) for (p, _), w in zip(candidates, weights)
    ) / float(weights.sum())
    target = probability > args.mask_threshold
    union = np.count_nonzero(target | local.astype(bool))
    agreement = float(np.count_nonzero(target & local.astype(bool)) / max(1, union))
    info = {"local_tracking_iou": agreement, "used": True}
    if seed.any() and (not target.any() or agreement < 0.1):
        info.update(used=False, reason="tracking_disagrees_with_positive_anchor")
        return local.copy(), local.astype(np.float32), info
    score = (1 - args.temporal_weight) * local + args.temporal_weight * probability
    result = (score > args.mask_threshold).astype(np.uint8)
    if local.any():
        support = cv2.dilate(
            local | seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        )
        result &= support
    # A positive annotation is not a negative instruction on its exterior.
    result |= positive_core(
        seed if trusted_seed is None else trusted_seed, args.core_radius
    )
    return result, score, info


def coordinate_boundaries(masks, source_masks, scores, locked_labels=()):
    """Joint evidence-based ownership only for new narrow boundary conflicts."""
    labels = sorted(masks)
    if len(labels) < 2:
        return masks, {"resolved_pixels": 0, "ambiguous_pixels": 0}
    occupancy = sum(m.astype(np.int16) for m in masks.values())
    original_count = (
        sum(m.astype(np.int16) for m in source_masks.values())
        if source_masks
        else np.zeros_like(occupancy)
    )
    editable = (occupancy > 1) & (original_count < 2)
    # Large label contradictions are semantic review cases, not edge cleanup.
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            overlap = masks[a] & masks[b]
            if (
                a in locked_labels
                or b in locked_labels
                or overlap.sum() > 0.5 * max(1, min(masks[a].sum(), masks[b].sum()))
            ):
                editable &= ~overlap.astype(bool)
    ranks = []
    for label in labels:
        source = source_masks.get(label, np.zeros_like(masks[label]))
        if source.any():
            distance = cv2.distanceTransform(1 - source, cv2.DIST_L2, 5)
            proximity = np.exp(-distance / 12)
        else:
            proximity = np.zeros_like(source, dtype=np.float32)
        rank = (
            scores.get(label, masks[label].astype(np.float32)) * 0.45
            + proximity * 0.45
            + source * 0.75
            + positive_core(source) * 0.75
        )
        ranks.append(np.where(masks[label].astype(bool), rank, -100))
    rank_array = np.stack(ranks)
    maximum = rank_array.max(axis=0)
    winners = rank_array >= maximum[None] - 0.03
    unique = winners.sum(axis=0) == 1
    resolved = editable & unique
    result = {
        label: mask & (~resolved | winners[i]).astype(np.uint8)
        for i, (label, mask) in enumerate((label, masks[label]) for label in labels)
    }
    return result, {
        "resolved_pixels": int(resolved.sum()),
        "ambiguous_pixels": int((editable & ~unique).sum()),
    }


def improve_frame_masks(doc, original_masks, temporal_inputs, args, frame_index=0):
    height, width = args.height, args.width
    masks = class_masks(doc["shapes"], height, width)
    before = {label: m.copy() for label, m in masks.items()}
    scores, evidence = {}, {}
    locked = {
        s["label"]
        for s in doc["shapes"]
        if (s.get("flags") or {}).get("preserve")
        or (s.get("flags") or {}).get("no_refine")
    }
    automatic = {
        s["label"] for s in doc["shapes"] if (s.get("flags") or {}).get("automatic")
    }
    removed = {r["shape_index"] for r in doc.get("surgslot_filtered_shapes", [])}
    # Conservatively lock a whole class if one of its source regions is excluded.
    # Class-level merging must not silently edit an explicitly protected polygon.
    locked.update(
        s["label"]
        for i, s in enumerate(doc.get("surgslot_original_shapes", doc["shapes"]))
        if i not in removed and not selected_for_refinement(s, frame_index, args)
    )
    trusted_masks = class_masks(
        [
            s
            for i, s in enumerate(doc.get("surgslot_original_shapes", doc["shapes"]))
            if i not in removed and not (s.get("flags") or {}).get("rough")
        ],
        height,
        width,
    )
    for label in set(masks) | set(temporal_inputs):
        local = masks.get(label, np.zeros((height, width), np.uint8))
        source = original_masks.get(label, np.zeros_like(local))
        trusted = trusted_masks.get(label, np.zeros_like(local))
        if label in locked:
            continue
        if temporal_inputs.get(label):
            candidates, anchor_ids = temporal_inputs[label]
            local, scores[label], evidence[label] = temporal_consensus(
                local, source, candidates, args, trusted
            )
            evidence[label]["anchors"] = anchor_ids
            automatic.add(label)
        if label in automatic:
            local = clean_candidate(local, source, label, args)
            local |= positive_core(trusted, args.core_radius)
        masks[label] = local
    coordination = {"resolved_pixels": 0, "ambiguous_pixels": 0}
    if args.coordinate_boundaries:
        masks, coordination = coordinate_boundaries(
            masks, original_masks, scores, locked
        )
    changed = {
        label
        for label, mask in masks.items()
        if label not in before or not np.array_equal(mask, before[label])
    }
    shapes = [s for s in doc["shapes"] if s["label"] not in changed]
    for label in sorted(changed):
        polygons, holes = mask_shapes(
            masks[label],
            label,
            args.min_area,
            args.polygon_epsilon,
            {
                "automatic": True,
                "temporal": bool(evidence.get(label, {}).get("used")),
                "refined": label in original_masks,
                "needs_review": False,
            },
        )
        for polygon in polygons:
            polygon["flags"]["needs_review"] = bool(holes)
            polygon["source_shape_indices"] = [
                i
                for i, s in enumerate(doc.get("surgslot_original_shapes", []))
                if s["label"] == label
            ]
        shapes.extend(polygons)
    doc["shapes"] = shapes
    # Report the masks that LabelMe will actually draw after polygon conversion.
    return class_masks(shapes, height, width), evidence, coordination, changed


def render_overlay(frame, masks, colors, conflicts, alpha, original_masks=None):
    # Average overlapping class colours, independent of iteration order.
    color_sum = np.zeros_like(frame, dtype=np.float32)
    count = np.zeros(frame.shape[:2], dtype=np.float32)
    for label, mask in masks.items():
        color_sum += mask[:, :, None] * np.array(colors[label], dtype=np.float32)
        count += mask
    active = count > 0
    overlay = frame.copy()
    overlay[active] = (
        frame[active] * (1 - alpha) + color_sum[active] / count[active, None] * alpha
    ).astype(np.uint8)
    for label, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, colors[label], 2)
    for row, label in enumerate(masks):
        x, y = max(12, frame.shape[1] - 200), 62 + row * 25
        cv2.rectangle(overlay, (x, y - 12), (x + 14, y + 2), colors[label], -1)
        cv2.putText(
            overlay,
            label,
            (x + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    yy, xx = np.indices(count.shape)
    for conflict in conflicts:
        a, b = conflict["labels"]
        overlap = masks[a] & masks[b]
        if original_masks is not None and a in original_masks and b in original_masks:
            overlap = overlap & (1 - (original_masks[a] & original_masks[b]))
        hatch = overlap.astype(bool) & ((xx + yy) % 14 < 2)
        overlay[hatch] = (255, 255, 255)
    return overlay


def export_results(
    frames, documents, anchors, tasks, work, output, visual, args, refinements=None
):
    refinements = refinements or []
    by_frame = {}
    for task in tasks:
        for span in task["spans"]:
            step = -1 if span["reverse"] else 1
            for index in range(task["anchor"] + step, span["end"] + step, step):
                by_frame.setdefault((index, task["label"]), []).append(
                    (task, span["reverse"])
                )
    temporal_by_frame = {}
    for task in tasks:
        for span in task.get("temporal_spans", []):
            step = -1 if span["reverse"] else 1
            for index in range(task["anchor"] + step, span["end"] + step, step):
                temporal_by_frame.setdefault((index, task["label"]), []).append(
                    (task, span["reverse"])
                )
    filtered_count = temporal_class_frames = boundary_resolved_pixels = 0
    local_fallback_count = 0
    final_automatic_count = final_propagated_count = 0
    colors = {
        label: tuple(
            int(c)
            for c in cv2.cvtColor(
                np.uint8([[[i * 179 // len(anchors), 190, 245]]]), cv2.COLOR_HSV2BGR
            )[0, 0]
        )
        for i, label in enumerate(anchors)
    }
    write_json(
        visual / "legend.json", {k: {"rgb": list(v[::-1])} for k, v in colors.items()}
    )
    width, height = args.width, args.height
    writer = cv2.VideoWriter(
        str(visual / "preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width + width % 2, height + height % 2),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV MP4 writer failed to open")
    reviews, generated, preserved, hole_count, source_count, replaced = (
        [],
        0,
        0,
        0,
        0,
        0,
    )
    try:
        for index, frame in enumerate(frames):
            doc = (
                copy.deepcopy(documents[index])
                if documents[index] is not None
                else {"version": "7.5.0", "flags": {}, "shapes": []}
            )
            raw_manual = copy.deepcopy(doc.get("shapes", []))
            manual = [copy.deepcopy(shape) for _, shape in valid_source_shapes(doc)]
            filtered = doc.get("surgslot_filtered_shapes", [])
            filtered_count += len(filtered)
            source_count += len(raw_manual)
            doc = apply_refinements(doc, index, refinements, work, args)
            frame_records = [r for r in refinements if r["frame"] == index]
            changed_records = [r for r in frame_records if r.get("export_changed")]
            replaced += len(changed_records)
            local_fallback_count += len(manual) - len(changed_records)
            manual_labels = {s["label"] for s in manual}
            automatic_labels = {
                s["label"] for s in doc["shapes"] if s.get("flags", {}).get("automatic")
            }
            evidence, reasons = {}, []
            for record in frame_records:
                if not record["accepted"]:
                    reasons.append(
                        {
                            "type": "refinement_rejected",
                            "label": record["label"],
                            "shape_index": record["shape_index"],
                            "reason": record["trials"][-1]["reason"],
                        }
                    )
                elif any(t["clipped_pixels"] for t in record["trials"]):
                    reasons.append(
                        {
                            "type": "refinement_clipped",
                            "label": record["label"],
                            "shape_index": record["shape_index"],
                        }
                    )
                if record.get("holes_split"):
                    reasons.append(
                        {
                            "type": "polygon_holes_split",
                            "label": record["label"],
                            "count": record["holes_split"],
                        }
                    )
                    hole_count += record["holes_split"]
            for label in anchors:
                if label in manual_labels:
                    continue
                entries = by_frame.get((index, label), [])
                if not entries:
                    raise RuntimeError(
                        f"Missing propagation: frame={index + 1}, label={label}"
                    )
                candidates = []
                for task, reverse in entries:
                    with np.load(prediction_path(work, task, reverse, index)) as data:
                        candidates.append(
                            (data["probability"], abs(index - task["anchor"]))
                        )
                mask, iou = fuse_predictions(candidates, args.mask_threshold)
                evidence[label] = {
                    "anchors": [t["anchor"] + 1 for t, _ in entries],
                    "bidirectional_iou": iou,
                    "empty": not bool(mask.any()),
                }
                if iou is not None and iou < args.disagreement_iou:
                    reasons.append(
                        {"type": "direction_disagreement", "label": label, "iou": iou}
                    )
                shapes, holes = mask_shapes(
                    mask,
                    label,
                    args.min_area,
                    args.polygon_epsilon,
                    {"automatic": True},
                    "SurgSLOT-SAM3 propagation; review required",
                )
                hole_count += holes
                if holes:
                    reasons.append(
                        {"type": "polygon_holes_split", "label": label, "count": holes}
                    )
                if shapes:
                    automatic_labels.add(label)
                    doc["shapes"].extend(shapes)
                    generated += len(shapes)
            masks = class_masks(doc["shapes"], height, width)
            original_masks = class_masks(manual, height, width)
            temporal_inputs = {}
            if args.refine and args.temporal_stride:
                for label in anchors:
                    entries = temporal_by_frame.get((index, label), [])
                    if not entries:
                        continue
                    candidates = []
                    for task, reverse in entries:
                        with np.load(
                            prediction_path(work, task, reverse, index, "temporal")
                        ) as data:
                            candidates.append(
                                (data["probability"], abs(index - task["anchor"]))
                            )
                    temporal_inputs[label] = (
                        candidates,
                        [t["anchor"] + 1 for t, _ in entries],
                    )
            temporal_info, coordination, adjusted_labels = (
                {},
                {"resolved_pixels": 0},
                set(),
            )
            if args.refine:
                masks, temporal_info, coordination, adjusted_labels = (
                    improve_frame_masks(
                        doc, original_masks, temporal_inputs, args, index
                    )
                )
                automatic_labels |= adjusted_labels
                temporal_class_frames += sum(
                    v.get("used", False) for v in temporal_info.values()
                )
                boundary_resolved_pixels += coordination["resolved_pixels"]
            preserved += sum(shape in doc["shapes"] for shape in manual)
            final_automatic_count += sum(
                bool(s.get("flags", {}).get("automatic")) for s in doc["shapes"]
            )
            final_propagated_count += sum(
                bool(s.get("flags", {}).get("automatic"))
                and not s.get("flags", {}).get("refined", False)
                for s in doc["shapes"]
            )
            conflicts, _ = overlap_pairs(masks, automatic_labels, original_masks)
            _, manual_overlaps = overlap_pairs(original_masks, set())
            if conflicts:
                reasons.append({"type": "automatic_class_overlap", "pairs": conflicts})
            conflict_labels = {label for pair in conflicts for label in pair["labels"]}
            for shape in doc["shapes"]:
                if shape.get("flags", {}).get("automatic"):
                    shape["flags"]["needs_review"] = (
                        shape["flags"].get("needs_review", False)
                        or shape["label"] in conflict_labels
                        or any(r.get("label") == shape["label"] for r in reasons)
                    )
            doc.update(
                {
                    "imagePath": frame.name,
                    "imageData": None,
                    "imageHeight": height,
                    "imageWidth": width,
                }
            )
            doc["surgslot"] = {
                "frame_index": index,
                "human_labels_authoritative": True,
                "boundary_policy": (
                    "refine_with_fallback" if args.refine else "preserve"
                ),
                "automatic": evidence,
                "temporal": temporal_info,
                "boundary_coordination": coordination,
                "source_raw": len(raw_manual),
                "source_valid": len(manual),
                "filtered_shapes": filtered,
                "review": reasons,
            }
            write_json(output / frame.with_suffix(".json").name, doc)
            shutil.copy2(frame, output / frame.name)
            image = cv2.imread(str(frame))
            if image is None:
                raise RuntimeError(f"Cannot read image: {frame}")
            overlay = render_overlay(
                image, masks, colors, conflicts, args.alpha, original_masks
            )
            cv2.putText(
                overlay,
                f"{index + 1}/{len(frames)} source={len(manual)} "
                f"filtered={len(filtered)} tracked={sum(v.get('used', False) for v in temporal_info.values())} "
                f"propagated={sum(s.get('flags', {}).get('automatic', False) and not s.get('flags', {}).get('refined', False) for s in doc['shapes'])} "
                f"review={len(reasons)}",
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            # All labels share this one overlay, including their original overlaps.
            if not cv2.imwrite(str(visual / f"{frame.stem}.jpg"), overlay):
                raise RuntimeError(f"Cannot write overlay for {frame}")
            writer.write(
                cv2.copyMakeBorder(
                    overlay, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT
                )
            )
            reviews.append(
                {
                    "frame": frame.name,
                    "manual_shapes": len(manual),
                    "raw_manual_shapes": len(raw_manual),
                    "filtered_shapes": filtered,
                    "temporal": temporal_info,
                    "boundary_coordination": coordination,
                    "refined_source_shapes": len(changed_records),
                    "automatic_shapes": sum(
                        s.get("flags", {}).get("automatic", False)
                        for s in doc["shapes"]
                    ),
                    "automatic": evidence,
                    "review": reasons,
                    "manual_overlaps": manual_overlaps,
                }
            )
    finally:
        writer.release()
    write_json(output / "review.json", reviews)
    return {
        "frames": len(frames),
        "original_shapes_recorded": source_count,
        "filtered_source_shapes": filtered_count,
        "valid_source_shapes": source_count - filtered_count,
        "local_refinement_fallback_shapes": local_fallback_count,
        "temporally_supported_class_frames": temporal_class_frames,
        "boundary_resolved_pixels": boundary_resolved_pixels,
        "original_shapes_preserved": preserved,
        "refined_source_shapes": replaced,
        "propagated_polygons": final_propagated_count,
        "generated_polygons": final_automatic_count,
        "frames_needing_review": sum(bool(r["review"]) for r in reviews),
        "automatic_holes_split": hole_count,
    }


def export_refinements(frames, documents, workers, work, directory, args):
    records = [r for w in workers for t in w["results"] for r in t["refinements"]]
    if not records:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    for index in sorted({r["frame"] for r in records}):
        doc = apply_refinements(documents[index], index, records, work, args)
        shapes = doc["shapes"]
        doc.update({"imagePath": frames[index].name, "imageData": None})
        write_json(directory / frames[index].with_suffix(".json").name, doc)
        shutil.copy2(frames[index], directory / frames[index].name)
        image, panels = cv2.imread(str(frames[index])), []
        for regions, title in [
            (documents[index]["shapes"], "Original"),
            (shapes, "Refined proposal"),
        ]:
            panel = image.copy()
            for shape in regions:
                contours, _ = cv2.findContours(
                    shape_mask(shape, args.height, args.width),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(panel, contours, -1, (0, 255, 255), 2)
            cv2.putText(
                panel, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            panels.append(panel)
        cv2.imwrite(
            str(directory / f"{frames[index].stem}_comparison.jpg"),
            np.concatenate(panels, axis=1),
        )
    write_json(directory / "refinement_report.json", records)
    return records


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    p.add_argument("--name", help="Output subfolder name; default is input folder name")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=SAM3_ROOT / "checkpoints/surgslot-sam3_iSurg.pt",
    )
    p.add_argument(
        "--gpus", default="0,1,2,3", help="CUDA-visible indices; one model per card"
    )
    p.add_argument("--fps", type=float, default=10)
    p.add_argument(
        "--image-size", type=int, default=672, help="Tracker input, multiple of 14"
    )
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--presence-threshold", type=float, default=0.5)
    p.add_argument(
        "--min-area", type=float, default=8, help="Minimum AUTOMATIC contour area"
    )
    p.add_argument(
        "--polygon-epsilon",
        type=float,
        default=1.0,
        help="Automatic contour tolerance, pixels",
    )
    p.add_argument("--disagreement-iou", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument(
        "--refine",
        action="store_true",
        default=True,
        help="Refine all human regions by default; filters can restrict selection",
    )
    p.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Legacy mode: preserve human boundaries",
    )
    p.add_argument(
        "--refine-label",
        action="append",
        default=[],
        help="Refine this label; repeatable",
    )
    p.add_argument(
        "--refine-frame",
        type=int,
        action="append",
        default=[],
        help="Limit refinement to 1-based frame positions",
    )
    p.add_argument("--box-padding", type=float, default=0.1)
    p.add_argument(
        "--prompt-dilate",
        type=float,
        default=5,
        help="Extra box padding in original pixels, reduced for tiny regions",
    )
    p.add_argument(
        "--refine-points",
        type=int,
        default=3,
        help="Distributed positive interior points",
    )
    p.add_argument("--refine-attempts", type=int, default=2)
    p.add_argument(
        "--max-expand-px",
        type=float,
        default=32,
        help="Maximum candidate edit distance outside original mask",
    )
    p.add_argument("--min-seed-coverage", type=float, default=0.8)
    p.add_argument(
        "--rough-seed-coverage",
        type=float,
        default=0.35,
        help="Coverage required for shapes explicitly flagged rough",
    )
    p.add_argument("--max-area-ratio", type=float, default=3)
    p.add_argument("--source-min-area", type=float, default=64)
    p.add_argument("--artifact-max-area", type=float, default=1024)
    p.add_argument("--artifact-max-size", type=float, default=80)
    p.add_argument(
        "--protect-small-class", action="append", default=["bubble", "blood vessel"]
    )
    p.add_argument(
        "--core-radius",
        type=float,
        default=6,
        help="Protect trusted positive interiors from SAM cracks",
    )
    p.add_argument(
        "--temporal-stride",
        type=int,
        default=8,
        help="Keyframe spacing for tracking already annotated classes; 0 disables",
    )
    p.add_argument("--temporal-weight", type=float, default=0.65)
    p.add_argument("--boundary-smooth", type=int, default=3)
    p.add_argument("--min-component-area", type=int, default=64)
    p.add_argument(
        "--no-boundary-coordination", dest="coordinate_boundaries", action="store_false"
    )
    p.add_argument(
        "--anchor-every",
        type=int,
        default=1,
        help="Evaluation only: hold out JSON except every Nth frame",
    )
    p.add_argument(
        "--keep-work", action="store_true", help="Keep intermediate probabilities"
    )
    args = p.parse_args(argv)
    if (
        args.fps <= 0
        or args.image_size < 112
        or args.image_size % 14
        or args.anchor_every < 1
    ):
        p.error(
            "fps and anchor-every must be positive; image-size >=112 and divisible by 14"
        )
    if (
        any(not 0 < x < 1 for x in [args.mask_threshold, args.presence_threshold])
        or not 0 <= args.alpha <= 1
    ):
        p.error("thresholds must be in (0,1), alpha in [0,1]")
    if (
        args.min_area < 0
        or args.polygon_epsilon < 0
        or args.box_padding < 0
        or not 0 <= args.disagreement_iou <= 1
    ):
        p.error("area, epsilon, padding must be nonnegative; disagreement-iou in [0,1]")
    if (args.refine_label or args.refine_frame) and not args.refine:
        p.error("refine-label/refine-frame require --refine")
    if (
        args.prompt_dilate < 0
        or args.max_expand_px < 0
        or args.max_area_ratio < 1
        or not 1 <= args.refine_points <= 8
        or not 1 <= args.refine_attempts <= 4
        or not 0 < args.min_seed_coverage <= 1
        or not 0 < args.rough_seed_coverage <= 1
    ):
        p.error("Invalid refinement extent, coverage, point count or attempt count")
    if (
        min(
            args.source_min_area,
            args.artifact_max_area,
            args.artifact_max_size,
            args.core_radius,
            args.temporal_stride,
            args.boundary_smooth,
            args.min_component_area,
        )
        < 0
        or not 0 <= args.temporal_weight <= 1
    ):
        p.error("Invalid source filtering or temporal consistency parameters")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.input, args.checkpoint = args.input.resolve(), args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    gpus = [int(x) for x in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("--gpus requires unique nonnegative CUDA indices")
    frames, documents, anchors, (args.height, args.width), hashes = read_sequence(
        args.input, args.anchor_every, args
    )
    if set(args.refine_label) - set(anchors):
        raise ValueError("Refinement label has no annotation anchors")
    if any(f < 1 or f > len(frames) for f in args.refine_frame):
        raise ValueError("refine-frame must be a 1-based position in sorted frames")
    tasks = make_tasks(anchors, documents, args)
    name = args.name or args.input.name
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("--name must be a single folder name")
    output = args.output_root.resolve() / name
    visual, refined = output.with_name(name + "_visual"), output.with_name(
        name + "_refined"
    )
    for directory in (output, visual, refined):
        if (
            directory == args.input
            or args.input in directory.parents
            or directory in args.input.parents
        ):
            raise ValueError("Output and doctor's source directories must be separate")
        if directory.exists():
            raise FileExistsError(
                f"Output exists; choose a different --name: {directory}"
            )
    output.mkdir(parents=True)
    visual.mkdir(parents=True)
    work = Path(tempfile.mkdtemp(prefix=".work_", dir=output))
    frame_dir = work / "frames"
    frame_dir.mkdir()
    for index, frame in enumerate(frames):
        target = frame_dir / f"{index:06d}.jpg"
        if frame.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            target.symlink_to(frame)  # Existing tracker requires integer stems.
        else:
            with Image.open(frame) as image:
                image.convert("RGB").save(target)
    started = time.monotonic()
    print(
        f"{len(frames)} frames, {len(anchors)} labels, {len(tasks)} anchor tasks; GPUs={gpus}",
        flush=True,
    )
    print(f"Output: {output}\nVisual: {visual}", flush=True)
    try:
        workers = launch_workers(gpus, tasks, documents, frame_dir, work, args)
        refinements = export_refinements(
            frames, documents, workers, work, refined, args
        )
        summary = export_results(
            frames, documents, anchors, tasks, work, output, visual, args, refinements
        )
        write_json(output / "refinement_report.json", refinements)
        for filename, checksum in hashes.items():
            if (
                hashlib.sha256((args.input / filename).read_bytes()).hexdigest()
                != checksum
            ):
                raise RuntimeError(f"Source JSON changed during run: {filename}")
        summary.update(
            {
                "input": str(args.input),
                "checkpoint": str(args.checkpoint),
                "gpus": gpus,
                "reverse_implementation": "reversed frame index view + unchanged forward SurgSLOT",
                "workers": workers,
                "fps": args.fps,
                "anchors_1based": {k: [i + 1 for i in v] for k, v in anchors.items()},
                "anchor_every": args.anchor_every,
                "source_json_sha256": hashes,
                "source_json_unchanged": True,
                "refinement_attempted": len(refinements),
                "refinement_accepted": sum(r["accepted"] for r in refinements),
                "refinement_rejected": sum(not r["accepted"] for r in refinements),
                "refined_shapes": sum(
                    r.get("export_changed", False) for r in refinements
                ),
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "parameters": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "limitations": [
                    "Only annotated interest labels; no semantic discovery",
                    "Human class labels are authoritative; anchor boundaries are refined with validated fallback",
                    "Polygon holes preserved via adjacent polygon decomposition; raw probabilities via --keep-work",
                    "Temporary components only, no persistent instance tracking; human review required",
                ],
            }
        )
        write_json(output / "run_report.json", summary)
        if not args.keep_work:
            shutil.rmtree(work)
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in [
                        "frames",
                        "original_shapes_preserved",
                        "generated_polygons",
                        "frames_needing_review",
                        "refined_shapes",
                        "elapsed_seconds",
                    ]
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except BaseException:
        write_json(
            output / "FAILED.json",
            {"traceback": traceback.format_exc(), "work": str(work)},
        )
        raise


if __name__ == "__main__":
    main()
