# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
SAM3Train: training adapter for the non-multiplex SAM3 tracker.

Inherits Sam3TrackerBase (the core VOS tracker) and:
  - Builds all components resolution-aware (672 or 1008)
  - Adds prepare_prompt_inputs / forward_tracking adapted for
    SAM2-style BatchedVideoDatapoint (no input adapters needed)
  - Loads official sam3.pt / sam3.1.pt checkpoints (remaps tracker.model.* keys)
  - Supports selective freezing via freeze_image_encoder and freeze_patterns

State dict keys are 1:1 with the official tracker checkpoint:
  backbone.*, transformer.*, maskmem_backbone.*, sam_mask_decoder.*, etc.

Usage in Hydra config:
  model:
    _target_: training.model.sam3_train.SAM3Train
    image_size: 672
    freeze_image_encoder: true
"""

import fnmatch
import logging
from typing import List, Optional

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr

from sam3.model.cls_memory_attention import ClsMemoryAttention, ClsMemoryAttentionLayer
from sam3.model.decoder import TransformerDecoderLayerv2, TransformerEncoderCrossAttention
from sam3.model.memory import CXBlock, SimpleFuser, SimpleMaskDownSampler, SimpleMaskEncoder
from sam3.model.model_misc import TransformerWrapper
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_tracker_base import Sam3TrackerBase, concat_points
from sam3.model.sam3_tracker_utils import get_next_point, sample_box_points
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam.transformer import RoPEAttention

from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.category_name_list import get_category_names

logger = logging.getLogger(__name__)


class _MLPDropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.dropout(self.act(layer(x))) if i < self.num_layers - 1 else layer(x)
        return x


class SAM3Train(Sam3TrackerBase):
    """
    Training wrapper for the non-multiplex SAM3 tracker.
    Accepts BatchedVideoDatapoint from the SAM2 training pipeline.
    """

    def __init__(
        self,
        # --- architecture ---
        image_size: int = 1008,
        num_maskmem: int = 7,
        # --- training strategy ---
        prob_to_use_pt_input_for_train: float = 0.0,
        prob_to_use_pt_input_for_eval: float = 0.0,
        prob_to_use_box_input_for_train: float = 0.0,
        prob_to_use_box_input_for_eval: float = 0.0,
        num_frames_to_correct_for_train: int = 1,
        num_frames_to_correct_for_eval: int = 1,
        rand_frames_to_correct_for_train: bool = False,
        rand_frames_to_correct_for_eval: bool = False,
        num_init_cond_frames_for_train: int = 1,
        num_init_cond_frames_for_eval: int = 1,
        rand_init_cond_frames_for_train: bool = True,
        rand_init_cond_frames_for_eval: bool = False,
        add_all_frames_to_correct_as_cond: bool = False,
        num_correction_pt_per_frame: int = 7,
        pt_sampling_for_eval: str = "center",
        prob_to_sample_from_gt_for_train: float = 0.0,
        # --- cls token ---
        use_cls_tokens: bool = True,
        use_cls_mem_attn: bool = False,
        use_tissue_text: bool = False,
        # --- long memory ---
        use_long_mem_tpos_enc: bool = False,
        num_init_long_mem_frames: int = 0,
        num_correction_pt_per_long_mem_frame: int = 2,
        # --- freeze ---
        freeze_image_encoder: bool = False,
        freeze_patterns: Optional[List[str]] = None,
        # --- checkpoint ---
        checkpoint_path: Optional[str] = None,
        # --- backbone compile ---
        compile_image_encoder: bool = False,
        use_fa3: bool = False,
        use_rope_real: bool = False,
        # --- eval ---
        forward_backbone_per_frame_for_eval: bool = False,
        prob_to_dropout_spatial_mem: float = 0.0,
    ):
        backbone_stride = 14
        feat_size = image_size // backbone_stride  # 672→48, 1008→72
        interpol_size = [int(image_size * 8 / 7)] * 2

        maskmem_backbone = _build_tracker_maskmem_backbone(
            precompute_resolution=image_size,
            interpol_size=interpol_size,
        )
        transformer = _build_tracker_transformer(
            feat_size=feat_size,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
            use_cls_mem_attn=use_cls_mem_attn,
        )
        backbone = _build_tracker_backbone(
            image_size=image_size,
            compile_mode="max-autotune" if compile_image_encoder else None,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
        )

        super().__init__(
            backbone=backbone,
            transformer=transformer,
            maskmem_backbone=maskmem_backbone,
            image_size=image_size,
            backbone_stride=backbone_stride,
            num_maskmem=num_maskmem,
            multimask_output_in_sam=True,
            multimask_output_for_tracking=True,
            multimask_min_pt_num=0,
            multimask_max_pt_num=1,
            max_cond_frames_in_attn=4,
            keep_first_cond_frame=True,
            non_overlap_masks_for_mem_enc=False,
            offload_output_to_cpu_for_eval=False,
            trim_past_non_cond_mem_for_eval=False,
            sam_mask_decoder_extra_args={
                "dynamic_multimask_via_stability": True,
                "dynamic_multimask_stability_delta": 0.05,
                "dynamic_multimask_stability_thresh": 0.98,
            },
            compile_all_components=False,
            use_memory_selection=False,
            use_long_mem_tpos_enc=use_long_mem_tpos_enc,
            use_cls_tokens=use_cls_tokens,
            prob_to_dropout_spatial_mem=prob_to_dropout_spatial_mem,
        )

        if use_cls_mem_attn:
            logger.info(
                "[SAM3Train] ClsMemoryAttention enabled: cls tokens are excluded from "
                "self-attention and only participate in cross-attention (memory attention)."
            )

        # ── long memory ──
        if use_long_mem_tpos_enc:
            assert num_init_long_mem_frames > 0, (
                "num_init_long_mem_frames must be > 0 when use_long_mem_tpos_enc is True"
            )
        self.num_init_long_mem_frames = num_init_long_mem_frames
        self.num_correction_pt_per_long_mem_frame = num_correction_pt_per_long_mem_frame

        # ── Training-specific attributes ──
        self.rng = np.random.default_rng(seed=42)
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        self.use_tissue_text = use_tissue_text
        # Attributes expected by Sam3TrackerBase internals with safe training defaults
        self.teacher_force_obj_scores_for_mem = False
        self.iter_use_prev_mask_pred = True

        if self.use_cls_tokens:
            self._init_category_text_embeddings()

        # ── Load checkpoint ──
        if checkpoint_path:
            self._load_sam3_checkpoint(checkpoint_path)

        # ── Freeze ──
        if freeze_image_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("[SAM3Train] Froze image encoder (self.backbone)")

        if freeze_patterns:
            self._apply_freeze_patterns(freeze_patterns)

    # ─────────────────────────────────────────────────────────
    #  Forward entry point
    # ─────────────────────────────────────────────────────────
    def forward(self, input: BatchedVideoDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
            if backbone_frozen:
                # Run ViT backbone without grad; conv_s0/s1 on mask decoder need grad
                backbone_out = self._forward_image_frozen_backbone(input.flat_img_batch)
            else:
                backbone_out = self.forward_image(input.flat_img_batch)
        else:
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}

        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)
        if self.use_cls_tokens:
            previous_stages_out = self._prepare_category_text_embeddings(previous_stages_out)
        return previous_stages_out

    def _init_category_text_embeddings(self):
        from transformers import AutoTokenizer, CLIPModel

        class_names = get_category_names(self.use_tissue_text)
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", from_tf=False)
        language_model = clip.text_model.eval()
        tokenized_input = tokenizer(class_names, padding=True, return_tensors="pt")
        with torch.no_grad():
            clip_text_features = language_model(**tokenized_input).pooler_output
        self.register_buffer("category_clip_text_features", clip_text_features)
        self.text_token_projection = _MLPDropout(
            language_model.config.hidden_size,
            2048,
            self.hidden_dim,
            3,
            dropout=0.2,
        )

    def _prepare_category_text_embeddings(self, previous_stages_out):
        text_features = self.text_token_projection(self.category_clip_text_features)
        for i in range(len(previous_stages_out)):
            previous_stages_out[i]["text_features"] = text_features
        return previous_stages_out

    def _forward_image_frozen_backbone(self, img_batch: torch.Tensor):
        """
        Run backbone under no_grad, then re-apply conv_s0/s1 with gradients.
        conv_s0 and conv_s1 live on sam_mask_decoder (trainable), not backbone.
        """
        with torch.no_grad():
            backbone_out = self.backbone.forward_image(img_batch)["sam2_backbone_out"]
        backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        for i in range(len(backbone_out["backbone_fpn"])):
            backbone_out["backbone_fpn"][i] = self._maybe_clone(backbone_out["backbone_fpn"][i])
        for i in range(len(backbone_out["vision_pos_enc"])):
            backbone_out["vision_pos_enc"][i] = self._maybe_clone(backbone_out["vision_pos_enc"][i])
        return backbone_out

    # ─────────────────────────────────────────────────────────
    #  Prompt preparation (adapted from SAM2Train)
    # ─────────────────────────────────────────────────────────
    def prepare_prompt_inputs(
        self, backbone_out, input: BatchedVideoDatapoint, start_frame_idx: int = 0
    ):
        """Prepare mask / point / box prompts from BatchedVideoDatapoint GT masks."""
        gt_masks_per_frame = {
            t: masks.unsqueeze(1)  # [O, 1, H, W]
            for t, masks in enumerate(input.masks)
        }
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames
        is_long_mem_batch = (
            bool(input.is_long_mem_batch[0].item())
            if hasattr(input, "is_long_mem_batch") and input.is_long_mem_batch is not None
            else False
        )

        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval

        if num_frames == 1:
            # static image mode: always use point input
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1

        assert num_init_cond_frames >= 1
        use_pt_input = self.rng.random() < prob_to_use_pt_input

        if rand_init_cond_frames and num_init_cond_frames > 1:
            num_init_cond_frames = self.rng.integers(1, num_init_cond_frames, endpoint=True)

        if use_pt_input and rand_frames_to_correct and num_frames_to_correct > num_init_cond_frames:
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )

        backbone_out["use_pt_input"] = use_pt_input

        if num_init_cond_frames == 1:
            if self.num_init_long_mem_frames > 0 and is_long_mem_batch and self.rng.random() < 0.5:
                # Add consecutive long-mem frames as extra conditioning frames
                init_cond_frames = [
                    start_frame_idx + t
                    for t in range(num_init_cond_frames + self.num_init_long_mem_frames)
                ]
            else:
                init_cond_frames = [start_frame_idx]
        else:
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()

        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]

        backbone_out["mask_inputs_per_frame"] = {}
        backbone_out["point_inputs_per_frame"] = {}

        for i, t in enumerate(init_cond_frames):
            # Long-memory frames always use point prompts, never masks or boxes.
            is_long_mem_frame = is_long_mem_batch and i in range(1, self.num_init_long_mem_frames + 1)
            if not use_pt_input and not is_long_mem_frame:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = (not is_long_mem_frame) and self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(gt_masks_per_frame[t])
                else:
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method="uniform" if self.training else self.pt_sampling_for_eval,
                    )
                backbone_out["point_inputs_per_frame"][t] = {
                    "point_coords": points,
                    "point_labels": labels,
                }

        if self.use_long_mem_tpos_enc:
            # All cond frames (including long-mem) get corrections so they attend properly
            frames_to_add_correction_pt = init_cond_frames
        elif not use_pt_input:
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = init_cond_frames + self.rng.choice(
                backbone_out["frames_not_in_init_cond"], extra_num, replace=False
            ).tolist()

        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt
        return backbone_out

    # ─────────────────────────────────────────────────────────
    #  Tracking loop (adapted from Sam3TrackerBase for BatchedVideoDatapoint)
    # ─────────────────────────────────────────────────────────
    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict: bool = False
    ):
        """Run tracking frame-by-frame, sampling correction clicks on the fly."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(
                backbone_out
            )

        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

        for stage_id in processing_order:
            # BatchedVideoDatapoint: flat_obj_to_img_idx[t] gives img indices for frame t
            img_ids = input.flat_obj_to_img_idx[stage_id]

            if img_feats_already_computed:
                current_image = input.flat_img_batch[img_ids]
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                (
                    current_image,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(input.flat_img_batch, img_ids)

            gt_masks = backbone_out["gt_masks_per_frame"].get(stage_id, None)
            current_out = self._track_step_with_corrections(
                stage_id=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                current_image=current_image,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=gt_masks,
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
            )

            add_as_cond = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_as_cond:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict

        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Remove obj_ptr to avoid DDP unused-param errors with activation checkpointing
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]
        return all_frame_outputs

    def _sam3_track_step(self, **kwargs):
        """Wraps parent track_step() and returns intermediate SAM outputs as explicit values,
        mirroring SAM2's _track_step() pattern."""
        current_out = super().track_step(**kwargs)
        pix_feat_with_mem = current_out.pop("_pix_feat_with_mem")
        high_res_features = current_out.pop("_high_res_features")
        sam_outputs = current_out.pop("_sam_outputs")
        out_cls_tokens = current_out.get("out_cls_tokens", None)
        return current_out, pix_feat_with_mem, high_res_features, sam_outputs, out_cls_tokens

    def _track_step_with_corrections(
        self,
        stage_id,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        current_image,
        point_inputs,
        mask_inputs,
        gt_masks,
        frames_to_add_correction_pt,
        output_dict,
        num_frames,
    ):
        """Run track_step then optionally add iterative correction clicks."""
        current_out, pix_feat_with_mem, high_res_features, sam_outputs, memory_cls_tokens = self._sam3_track_step(
            frame_idx=stage_id,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=current_image,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=num_frames,
        )
        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            _,
            object_score_logits,
            _,
        ) = sam_outputs

        current_out["is_init_cond_frame"] = is_init_cond_frame
        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]
        current_out["multistep_out_cls_tokens"] = [memory_cls_tokens]

        if stage_id in frames_to_add_correction_pt and gt_masks is not None:
            all_pred_masks = [low_res_masks]
            all_pred_high_res_masks = [high_res_masks]
            all_pred_multimasks = [low_res_multimasks]
            all_pred_high_res_multimasks = [high_res_multimasks]
            all_pred_ious = [ious]
            all_point_inputs = [point_inputs]
            all_object_score_logits = [object_score_logits]
            all_cls_tokens = [memory_cls_tokens]

            # long-mem init frames (non-reference cond frames) use fewer correction points
            if self.use_long_mem_tpos_enc and stage_id != frames_to_add_correction_pt[0]:
                num_corr = self.num_correction_pt_per_long_mem_frame
            else:
                num_corr = self.num_correction_pt_per_frame

            for _ in range(num_corr):
                sample_from_gt = (
                    self.training
                    and self.prob_to_sample_from_gt_for_train > 0
                    and self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
                pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
                new_points, new_labels = get_next_point(
                    gt_masks=gt_masks,
                    pred_masks=pred_for_new_pt,
                    method="uniform" if self.training else self.pt_sampling_for_eval,
                )
                point_inputs = concat_points(point_inputs, new_points, new_labels)
                multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=low_res_masks,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    memory_cls_tokens=memory_cls_tokens,
                )
                (
                    low_res_multimasks,
                    high_res_multimasks,
                    ious,
                    low_res_masks,
                    high_res_masks,
                    _,
                    object_score_logits,
                    memory_cls_tokens,  # updated out_cls_tokens from this correction step
                ) = sam_outputs
                all_pred_masks.append(low_res_masks)
                all_pred_high_res_masks.append(high_res_masks)
                all_pred_multimasks.append(low_res_multimasks)
                all_pred_high_res_multimasks.append(high_res_multimasks)
                all_pred_ious.append(ious)
                all_point_inputs.append(point_inputs)
                all_object_score_logits.append(object_score_logits)
                all_cls_tokens.append(memory_cls_tokens)

            current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
            current_out["multistep_pred_masks_high_res"] = torch.cat(all_pred_high_res_masks, dim=1)
            current_out["multistep_pred_multimasks"] = all_pred_multimasks
            current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
            current_out["multistep_pred_ious"] = all_pred_ious
            current_out["multistep_point_inputs"] = all_point_inputs
            current_out["multistep_object_score_logits"] = all_object_score_logits
            current_out["multistep_out_cls_tokens"] = all_cls_tokens
            current_out["pred_masks"] = low_res_masks
            current_out["pred_masks_high_res"] = high_res_masks

            # Re-encode memory with the corrected (final) mask, matching SAM2 behavior
            # where memory is committed after all correction clicks.
            if self.num_maskmem > 0:
                maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                    image=current_image,
                    current_vision_feats=current_vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=high_res_masks,
                    object_score_logits=object_score_logits,
                    is_mask_from_pts=(point_inputs is not None),
                    output_dict=output_dict,
                    is_init_cond_frame=is_init_cond_frame,
                )
                current_out["maskmem_features"] = maskmem_features
                current_out["maskmem_pos_enc"] = maskmem_pos_enc

        return current_out

    # ─────────────────────────────────────────────────────────
    #  Checkpoint loading (same key remapping as multiplex)
    # ─────────────────────────────────────────────────────────
    def _load_sam3_checkpoint(self, ckpt_path: str):
        logger.info(f"[SAM3Train] Loading checkpoint from {ckpt_path}")
        with g_pathmgr.open(ckpt_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        has_tracker_prefix = any(k.startswith("tracker.") for k in ckpt)
        if has_tracker_prefix:
            remapped = {}
            tracker_prefix = "tracker."
            det_backbone_prefix = "detector.backbone."
            for k, v in ckpt.items():
                if k.startswith(tracker_prefix):
                    remapped[k[len(tracker_prefix):]] = v
                elif k.startswith(det_backbone_prefix):
                    new_key = "backbone." + k[len(det_backbone_prefix):]
                    if new_key not in remapped:
                        remapped[new_key] = v
            ckpt = remapped
            print(f"[SAM3Train] Remapped checkpoint: {len(ckpt)} keys")

        model_state = self.state_dict()
        filtered_ckpt, skipped = {}, []
        for k, v in ckpt.items():
            if k in model_state and v.shape != model_state[k].shape:
                skipped.append(f"{k}: ckpt {v.shape} vs model {model_state[k].shape}")
            else:
                filtered_ckpt[k] = v
        if skipped:
            print(f"[SAM3Train] Skipped {len(skipped)} shape-mismatched keys:")
            for s in skipped:
                print(f"  {s}")

        missing, unexpected = self.load_state_dict(filtered_ckpt, strict=False)
        print(f"[SAM3Train] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"[SAM3Train] first 10 missing: {missing[:10]}")
        if unexpected:
            print(f"[SAM3Train] first 10 unexpected: {unexpected[:10]}")

    # ─────────────────────────────────────────────────────────
    #  Freeze helpers
    # ─────────────────────────────────────────────────────────
    def _apply_freeze_patterns(self, freeze_patterns: List[str]):
        n_frozen = 0
        for name, p in self.named_parameters():
            if any(fnmatch.fnmatchcase(name, pat) for pat in freeze_patterns if pat):
                p.requires_grad = False
                n_frozen += 1
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[SAM3Train] freeze_patterns froze {n_frozen} params "
            f"(trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M = {100*trainable/total:.1f}%)"
        )


# ═══════════════════════════════════════════════════════════
#  Resolution-aware builder functions
# ═══════════════════════════════════════════════════════════

def _build_tracker_maskmem_backbone(
    precompute_resolution: int = 1008,
    interpol_size: list = None,
):
    """Memory encoder for the non-multiplex tracker (out_dim=64, mem_dim=64)."""
    if interpol_size is None:
        interpol_size = [int(precompute_resolution * 8 / 7)] * 2

    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,  # matches out_dim=64
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3,
        stride=2,
        padding=1,
        interpol_size=interpol_size,
    )
    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)
    return SimpleMaskEncoder(
        out_dim=64,  # non-multiplex uses 64 (multiplex uses 256)
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


def _build_tracker_transformer(
    feat_size: int = 72,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    use_cls_mem_attn: bool = False,
):
    """Memory attention transformer for the non-multiplex tracker.
    Uses RoPEAttention with kv_in_dim=64 to match mem_dim=64.

    When use_cls_mem_attn=True, builds ClsMemoryAttention so that cls tokens
    are excluded from self-attention but still participate in cross-attention.
    Otherwise falls back to the standard TransformerEncoderCrossAttention.
    """
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[feat_size, feat_size],
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,  # matches mem_dim=64
        rope_theta=10000.0,
        feat_sizes=[feat_size, feat_size],
        rope_k_repeat=True,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    layer_cls = ClsMemoryAttentionLayer if use_cls_mem_attn else TransformerDecoderLayerv2
    encoder_layer = layer_cls(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )
    encoder_cls = ClsMemoryAttention if use_cls_mem_attn else TransformerEncoderCrossAttention
    encoder = encoder_cls(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )
    return TransformerWrapper(encoder=encoder, decoder=None, d_model=256)


def _build_tracker_backbone(
    image_size: int = 1008,
    compile_mode=None,
    use_fa3: bool = False,
    use_rope_real: bool = False,
):
    """Vision backbone: SAM3VLBackbone (text=None) wrapping Sam3DualViTDetNeck + ViT.
    Sam3TrackerBase.forward_image() expects backbone.forward_image()["sam2_backbone_out"]."""
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=image_size,
    )
    vit_backbone = ViT(
        img_size=image_size,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    dual_neck = Sam3DualViTDetNeck(
        trunk=vit_backbone,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        add_sam2_neck=True,  # required: Sam3TrackerBase.forward_image uses sam2_backbone_out
    )
    # scalp=1: drop the lowest-resolution (0.5x) output before building backbone_fpn
    return SAM3VLBackbone(scalp=1, visual=dual_neck, text=None)


def _load_sam3_checkpoint_into(model, ckpt_path: str):
    """Standalone checkpoint loader; works on any Sam3TrackerBase subclass."""
    from iopath.common.file_io import g_pathmgr
    logger.info(f"[SAM3] Loading checkpoint from {ckpt_path}")
    with g_pathmgr.open(ckpt_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    has_tracker_prefix = any(k.startswith("tracker.") for k in ckpt)
    if has_tracker_prefix:
        remapped = {}
        for k, v in ckpt.items():
            if k.startswith("tracker."):
                remapped[k[len("tracker."):]] = v
            elif k.startswith("detector.backbone."):
                new_key = "backbone." + k[len("detector.backbone."):]
                if new_key not in remapped:
                    remapped[new_key] = v
        ckpt = remapped
        print(f"[SAM3] Remapped checkpoint: {len(ckpt)} keys")

    model_state = model.state_dict()
    filtered, skipped = {}, []
    for k, v in ckpt.items():
        if k in model_state and v.shape != model_state[k].shape:
            skipped.append(f"{k}: ckpt {v.shape} vs model {model_state[k].shape}")
        else:
            filtered[k] = v
    if skipped:
        print(f"[SAM3] Skipped {len(skipped)} shape-mismatched keys:")
        for s in skipped:
            print(f"  {s}")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[SAM3] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[SAM3] first 10 missing: {missing[:10]}")
    if unexpected:
        print(f"[SAM3] first 10 unexpected: {unexpected[:10]}")


def build_sam3_predictor(
    checkpoint_path: str,
    image_size: int = 1008,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    apply_postprocessing: bool = False,
    apply_long_mem: bool = False,
    num_long_mem_frame: int = 3,
    num_long_mem_candidate: int = 5,
    long_mem_iou_thr: float = 0.95,
    use_cls_inference: bool = False,
    use_cls_mem_attn: bool = False,
    use_long_mem_tpos_enc: bool = False,
    use_continuous_divemem: bool = False,
    proto_update_thr: float = 0.95,
    ema_alpha_proto: float = 0.95,
    short_mem_frames: int = 6,
    memory_stride: int = 1,
    training_config_file: str = "",
):
    """
    Build Sam3TrackerPredictor with the same architecture as SAM3Train and load
    a fine-tuned checkpoint. Use this for inference with vos_inference.py.
    """
    from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor

    backbone_stride = 14
    feat_size = image_size // backbone_stride
    interpol_size = [int(image_size * 8 / 7)] * 2

    maskmem_backbone = _build_tracker_maskmem_backbone(
        precompute_resolution=image_size,
        interpol_size=interpol_size,
    )
    transformer = _build_tracker_transformer(
        feat_size=feat_size,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        use_cls_mem_attn=use_cls_mem_attn,
    )
    backbone = _build_tracker_backbone(
        image_size=image_size,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )

    predictor = Sam3TrackerPredictor(
        backbone=backbone,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        apply_postprocessing=apply_postprocessing,
        apply_long_mem=apply_long_mem,
        num_long_mem_frame=num_long_mem_frame,
        num_long_mem_candidate=num_long_mem_candidate,
        long_mem_iou_thr=long_mem_iou_thr,
        use_cls_inference=use_cls_inference,
        use_long_mem_tpos_enc=use_long_mem_tpos_enc,
        use_continuous_divemem=use_continuous_divemem,
        proto_update_thr=proto_update_thr,
        ema_alpha_proto=ema_alpha_proto,
        short_mem_frames=short_mem_frames,
        memory_stride=memory_stride,
        training_config_file=training_config_file,
        image_size=image_size,
        backbone_stride=backbone_stride,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        max_cond_frames_in_attn=4,
        non_overlap_masks_for_mem_enc=False,
        offload_output_to_cpu_for_eval=False,
        trim_past_non_cond_mem_for_eval=False,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        compile_all_components=False,
    )

    print(
        "[SAM3 VOS] "
        f"apply_long_mem={apply_long_mem}, "
        f"num_long_mem_frame={num_long_mem_frame}, "
        f"num_long_mem_candidate={num_long_mem_candidate}, "
        f"long_mem_iou_thr={long_mem_iou_thr}, "
        f"use_cls_inference={use_cls_inference}, "
        f"use_continuous_divemem={use_continuous_divemem}, "
        f"proto_update_thr={proto_update_thr}, "
        f"ema_alpha_proto={ema_alpha_proto}, "
        f"short_mem_frames={short_mem_frames}, "
        f"memory_stride={memory_stride}"
    )

    _load_sam3_checkpoint_into(predictor, checkpoint_path)
    return predictor
