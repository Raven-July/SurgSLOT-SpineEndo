# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import gc
import json
from PIL import Image

from sam2.build_sam import build_surgslot_video_predictor
from sav_dataset.utils.endo_sav_benchmark import benchmark

from sam2.modeling.sam2_utils import get_next_point, sample_box_points

from sam2.utils.misc import concat_points
# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    with Image.open(path) as mask_img:
        palette = mask_img.getpalette()
        mask = np.array(mask_img).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if not os.path.exists(input_mask_path):
            pass
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    output_per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not output_per_obj_png_file:
        os.makedirs(os.path.join(output_mask_dir, video_name, 'all'), exist_ok=True)
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, 'all', f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask[output_mask>0] = object_id
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_separate_inference_per_object(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    save_frame_interval=1,
    prompt_type='mask',
    num_pts=1,
    offload_video_to_cpu=True,
):
    """
    Run VOS inference on a single video with the given predictor.

    Unlike `vos_inference`, this function run inference separately for each object
    in a video, which could be applied to datasets like LVOS or YouTube-VOS that
    don't have all objects to track appearing in the first frame (i.e. some objects
    might appear only later in the video).
    """
    # load the video frames and initialize the inference state on this video
    video_dir = os.path.join(base_video_dir, video_name)
    all_frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png']
    ]
    # The release inference path processes every frame.
    frame_names = sorted(all_frame_names, key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False,
        offload_video_to_cpu=offload_video_to_cpu,
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    for idx, name in enumerate(frame_names):
        if os.path.exists(os.path.join(input_mask_dir, video_name, f"{name}.png")):
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[idx],
                per_obj_png_file=False,  # our dataset combines all object masks into a single PNG file
                allow_missing=False,
            )
            for object_id, object_mask in per_obj_input_mask.items():
                # skip empty masks
                if not np.any(object_mask):
                    continue
                # if `use_all_masks=False`, we only use the first mask for each object
                if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                    continue
                print(f"adding mask from frame {idx} as input for {object_id=}")
                inputs_per_object[object_id][idx] = object_mask

    # step 1: run inference together for each object appearing in the first frame
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)
    # find the object appear in the first frame
    first_frame_object_ids = []
    latter_frame_object_ids = []
    for object_id in object_ids:
        if inputs_per_object[object_id].keys().__contains__(0):
            first_frame_object_ids.append(object_id)
        else:
            latter_frame_object_ids.append(object_id)

    if len(first_frame_object_ids) != 0:
        input_frame_idx = 0
        for object_id in first_frame_object_ids:
            if prompt_type == 'mask':
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    mask=inputs_per_object[object_id][input_frame_idx],
                )
            elif prompt_type == 'point':
                this_high_res_masks = None
                gt_mask = torch.from_numpy(inputs_per_object[object_id][input_frame_idx]).unsqueeze(0).unsqueeze(0).to(inference_state['device'])
                for _ in range(num_pts):
                    new_points, new_labels = get_next_point(
                        gt_masks=gt_mask,
                        pred_masks=(this_high_res_masks > score_thresh) if this_high_res_masks is not None else None,
                        method="center",
                    )
                    _, _, high_res_masks = predictor.add_new_points(
                        inference_state=inference_state,
                        frame_idx=input_frame_idx,
                        obj_id=object_id,
                        points=new_points,
                        labels=new_labels,
                        clear_old_points=False,
                    )
                    this_high_res_masks = high_res_masks[inference_state['obj_id_to_idx'][object_id]].unsqueeze(0)
            elif prompt_type == 'box':
                gt_mask = torch.from_numpy(inputs_per_object[object_id][input_frame_idx]).unsqueeze(0).unsqueeze(0).to(inference_state['device'])
                points, labels = sample_box_points(
                    gt_mask, noise=0
                )
                _, _, high_res_masks = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                    clear_old_points=True,
                )
                # # sample_box_points
                # input_mask = inputs_per_object[object_id][input_frame_idx]
                # y_indices, x_indices = np.where(input_mask > 0.5)
                # bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
                # pass
                # # Expand by five pixels.
                # bbox = [max(0, bbox[0] - 5), max(0, bbox[1] - 5), min(width - 1, bbox[2] + 5), min(height - 1, bbox[3] + 5)]
                # _, _, this_high_res_masks = predictor.add_new_points_or_box(
                #     inference_state=inference_state,
                #     frame_idx=input_frame_idx,
                #     obj_id=object_id,
                #     box=bbox,
                # )
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")


        # run propagation throughout the video and collect the results in a dict
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=input_frame_idx, reverse=False,
        ):
            obj_scores = out_mask_logits
            for i, out_obj_id in enumerate(out_obj_ids):
                output_scores_per_object[out_obj_id][out_frame_idx] = obj_scores[i: i + 1].to('cpu', non_blocking=True)

    # step 2: run inference separately for the object appearing in the latter frame
    for object_id in latter_frame_object_ids:
        # add those input masks to SAM 2 inference state before propagation
        input_frame_inds = sorted(inputs_per_object[object_id])
        predictor.reset_state(inference_state)
        for input_frame_idx in input_frame_inds:
            if prompt_type == 'mask':
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    mask=inputs_per_object[object_id][input_frame_idx],
                )
            elif prompt_type == 'point':
                this_high_res_masks = None
                gt_mask = torch.from_numpy(inputs_per_object[object_id][input_frame_idx]).unsqueeze(0).unsqueeze(0).to(
                    inference_state['device'])
                for _ in range(num_pts):
                    new_points, new_labels = get_next_point(
                        gt_masks=gt_mask,
                        pred_masks=(this_high_res_masks > score_thresh) if this_high_res_masks is not None else None,
                        method="center",
                    )
                    _, _, high_res_masks = predictor.add_new_points(
                        inference_state=inference_state,
                        frame_idx=input_frame_idx,
                        obj_id=object_id,
                        points=new_points,
                        labels=new_labels,
                        clear_old_points=False,
                    )
                    this_high_res_masks = high_res_masks[inference_state['obj_id_to_idx'][object_id]].unsqueeze(0)
            elif prompt_type == 'box':
                gt_mask = torch.from_numpy(inputs_per_object[object_id][input_frame_idx]).unsqueeze(0).unsqueeze(0).to(inference_state['device'])
                points, labels = sample_box_points(
                    gt_mask, noise=0
                )
                _, _, high_res_masks = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                    clear_old_points=True,
                )
                # input_mask = inputs_per_object[object_id][input_frame_idx]
                # y_indices, x_indices = np.where(input_mask > 0)
                # bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
                # _, _, high_res_masks = predictor.add_new_points_or_box(
                #     inference_state=inference_state,
                #     frame_idx=input_frame_idx,
                #     obj_id=object_id,
                #     box=bbox,
                # )
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")

        # run propagation throughout the video and collect the results in a dict
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min(input_frame_inds),
            reverse=False,
        ):
            obj_scores = out_mask_logits
            output_scores_per_object[object_id][out_frame_idx] = obj_scores.to('cpu', non_blocking=True)

        # # TODO: Add reverse inference.
        # # run propagation throughout the video and collect the results in a dict
        # for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
        #         inference_state,
        #         start_frame_idx=min(input_frame_inds),
        #         reverse=True,
        # ):
        #     obj_scores = out_mask_logits
        #     output_scores_per_object[object_id][out_frame_idx] = obj_scores.cpu().numpy()

    video_segments = {}
    # save_frame_interval controls the output frequency.
    for out_frame_idx in range(len(frame_names)):
        frame_time = int(frame_names[out_frame_idx])
        if frame_time % save_frame_interval != 0:
            continue
        video_segments[out_frame_idx] = {}
        for object_id in object_ids:
            if output_scores_per_object[object_id].keys().__contains__(out_frame_idx):
                this_mask_cuda = output_scores_per_object[object_id][out_frame_idx].to(inference_state['device'])
                video_segments[out_frame_idx][object_id] = (
                    (this_mask_cuda > score_thresh).cpu().numpy()
                )

    # step 3: save the output masks as per-object PNG files
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            per_obj_output_mask=per_obj_output_mask,
            frame_name=frame_names[out_frame_idx],
            output_per_obj_png_file=True,
            height=height,
            width=width,
            output_palette=input_palette,
        )

    # 1. Explicitly delete the SAM 2 state dictionary and large CPU tensor cache.
    del inference_state
    del output_scores_per_object
    del video_segments
    # 2. Force garbage collection and release cached GPU memory.
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/surgslot/surgslot_hiera_b+_512.yaml",
        help="SurgSLOT model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        required=True,
        help="path to the SurgSLOT checkpoint",
    )
    parser.add_argument(
        "--dataset_list_json",
        type=str,
        default=None,
        help="JSON file with a list of dataset configurations containing base_video_dir, "
             "input_mask_dir, output_mask_dir, gt_root, and an optional benchmark_type. "
             "Use it to run all datasets in one process (model loaded only once). "
             "When provided, the individual --base_video_dir/--input_mask_dir/--output_mask_dir/--gt_root args are ignored.",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        default=None,
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        default=None,
        help="directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run VOS prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        default=None,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--proto_update_thr",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--ema_alpha_proto",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="whether to use all available PNG files in input_mask_dir "
        "(default without this flag: just the first PNG file as input to the SAM 2 model; "
        "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--apply_long_mem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to apply long term memory",
    )
    parser.add_argument(
        "--num_long_mem_frame",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_long_mem_candidate",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--long_mem_iou_thr",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--use_cls_inference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use_continuous_divemem",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--short_term_frame_num",
        type=int,
        default=6,
        help="number of most-recent short-term memory frames to attend to (default: 6)",
    )
    parser.add_argument(
        "--debug_long_mem",
        action="store_true",
        help="print long-term memory selection and eviction details",
    )
    parser.add_argument(
        "--gt_root",
        default=None,
        help="Path to the GT folder. For SA-V, it's sav_val/Annotations_6fps or sav_test/Annotations_6fps",
    )
    parser.add_argument(
        "--benchmark_type",
        choices=("generic", "endovis2018"),
        default="generic",
        help="evaluation protocol used when a single dataset is passed",
    )
    parser.add_argument(
        "--gpu_id",
        type=int, default=0,
    )
    parser.add_argument(
        "--save_frame_interval",
        type=int, default=1,
    )

    parser.add_argument(
        "--keep_video_on_gpu",
        action="store_true",
        help="keep decoded video frames on GPU instead of offloading them to CPU",
    )
    # ----
    parser.add_argument(
        "-n", "--num_processes", default=8, type=int, help="Number of concurrent processes"
    )
    parser.add_argument(
        "-s",
        "--strict",
        help="Make sure every video in the gt_root folder has a corresponding video in the prediction",
        action="store_true",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Quietly run evaluation without printing the information out",
        action="store_true",
    )
    parser.add_argument(
        "--do_not_skip_first_and_last_frame",
        help="In SA-V val and test, we skip the first and the last annotated frames in evaluation. "
             "Set this to true for evaluation on settings that doen't skip first and last frames",
        action="store_true",
    )
    parser.add_argument(
        "--prompt_type", type=str, default='mask',
        help="prompt type, mask, point or box",
    )
    parser.add_argument(
        "--num_pts",
        type=int,
        default=1,
        help="number of points to use for point prompts (default: 1, which is the center point)",
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    print(f"Using GPU {args.gpu_id}")

    predictor = build_surgslot_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        apply_long_mem=args.apply_long_mem,
        num_long_mem_frame=args.num_long_mem_frame,
        num_long_mem_candidate=args.num_long_mem_candidate,
        long_mem_iou_thr=args.long_mem_iou_thr,
        hydra_overrides_extra=[],
        strict_loading=False,
        use_cls_inference=args.use_cls_inference,
        use_continuous_divemem=args.use_continuous_divemem,
        proto_update_thr=args.proto_update_thr,
        ema_alpha_proto=args.ema_alpha_proto,
        short_mem_frames=args.short_term_frame_num,
        debug_long_mem=args.debug_long_mem,
    )

    # Build the list of dataset configs to run (model is loaded only once above)
    if args.dataset_list_json is not None:
        with open(args.dataset_list_json) as f:
            dataset_configs = json.load(f)
    else:
        if args.base_video_dir is None or args.input_mask_dir is None \
                or args.output_mask_dir is None or args.gt_root is None:
            parser.error(
                "When --dataset_list_json is not provided, "
                "--base_video_dir, --input_mask_dir, --output_mask_dir, and --gt_root are all required."
            )
        dataset_configs = [{
            "base_video_dir": args.base_video_dir,
            "input_mask_dir": args.input_mask_dir,
            "output_mask_dir": args.output_mask_dir,
            "gt_root": args.gt_root,
            "benchmark_type": args.benchmark_type,
        }]

    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print("using only the first frame's mask in input_mask_dir as input to the SAM 2 model")

    epoch = os.path.basename(args.sam2_checkpoint).split('_')[-1].split('.')[0]
    epoch = 0 if not epoch.isdigit() else int(epoch)

    # Phase 1: inference (GPU) — loop over all datasets with model in memory
    for ds_cfg in dataset_configs:
        base_video_dir = ds_cfg["base_video_dir"]
        input_mask_dir = ds_cfg["input_mask_dir"]
        output_mask_dir = ds_cfg["output_mask_dir"]

        if args.video_list_file is not None:
            with open(args.video_list_file, "r") as f:
                video_names = [v.strip() for v in f.readlines()]
        else:
            video_names = sorted([
                p for p in os.listdir(base_video_dir)
                if os.path.isdir(os.path.join(base_video_dir, p))
            ])
        print(f"running VOS prediction on {len(video_names)} videos in {base_video_dir}:\n{video_names}")

        for n_video, video_name in enumerate(video_names):
            print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
            vos_separate_inference_per_object(
                predictor=predictor,
                base_video_dir=base_video_dir,
                input_mask_dir=input_mask_dir,
                output_mask_dir=output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                use_all_masks=args.use_all_masks,
                save_frame_interval=args.save_frame_interval,
                prompt_type=args.prompt_type,
                num_pts=args.num_pts,
                offload_video_to_cpu=not args.keep_video_on_gpu,
            )

        print(f"completed VOS prediction on {len(video_names)} videos -- output masks saved to {output_mask_dir}")

    # Phase 2: release GPU before CPU-only benchmark
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory released, starting benchmark...")

    # Phase 3: benchmark (CPU only)
    for ds_cfg in dataset_configs:
        output_mask_dir = ds_cfg["output_mask_dir"]
        gt_root = ds_cfg["gt_root"]
        benchmark_type = ds_cfg.get("benchmark_type", args.benchmark_type)
        if benchmark_type == "endovis2018":
            benchmark_endovis2018(args, gt_root, output_mask_dir, epoch=epoch)
        else:
            benchmark(
                [gt_root],
                [output_mask_dir],
                args.strict,
                args.num_processes,
                verbose=not args.quiet,
                skip_first_and_last=not args.do_not_skip_first_and_last_frame,
                epoch=epoch,
            )


def benchmark_endovis2018(args, gt_root, output_mask_dir, epoch=0):
    evaluate_object_id_groups = {
        'instrument': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'tissue': [11, 12, 17]
    }
    for object_id_group_name, evaluate_object_id_list in evaluate_object_id_groups.items():
        print(f'Running evaluation for {object_id_group_name}')
        benchmark(
            [gt_root],
            [output_mask_dir],
            strict=False,
            num_processes=args.num_processes,
            verbose=not args.quiet,
            skip_first_and_last=not args.do_not_skip_first_and_last_frame,
            evaluate_object_id_list=evaluate_object_id_list,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()
