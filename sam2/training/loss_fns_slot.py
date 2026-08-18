# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
    pixel_weights=None,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if pixel_weights is not None:
        loss = pixel_weights * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects

def compute_boundary_entropy_weights(pred_masks, gt_masks, kernel_size=9, alpha=0.5):
    # edge = compute_edge(y_hard)  # sobel or laplacian
    dilated = F.max_pool2d(gt_masks, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    eroded = -F.max_pool2d(-gt_masks, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    border_mask = (dilated - eroded).detach()

    prob_pred_masks = torch.sigmoid(pred_masks)
    prob_pred_masks = prob_pred_masks.clamp(min=1e-7, max=1 - 1e-7)
    entropy = -(prob_pred_masks * prob_pred_masks.log() + (1 - prob_pred_masks) * (1 - prob_pred_masks).log())
    entropy_norm = (entropy / math.log(2)).detach()  # Normalize to [0, 1].

    # print("entropy_norm max value {}, min value {}".format(entropy_norm.max(), entropy_norm.min()))
    # Downweight high-entropy pixels inside the boundary band; learn normally outside it.
    noise_score = entropy_norm * border_mask
    pixel_weight = 1.0 - alpha * noise_score
    pixel_weight = pixel_weight.clamp(min=0.1)
    return pixel_weight

class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        use_cls_tokens=False,
        use_entropy_weighted_loss=False,
        entropy_boundary_kernel_size=9,  # Odd dilation/erosion kernel size; defaults to 9.
        entropy_weighted_alpha=0.5,
        cls_pretrained_epoch=0,
        confidence_cls=0.7,
        ema_alpha_proto=0.95,
        vlc_temperature=0.01,
        compute_consistency_loss=False,
        repulsion_iou_thr=0.0,
        repulsion_margin=0.0,
        consistency_margin=0.7,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.use_cls_tokens = use_cls_tokens
        self.use_entropy_weighted_loss = use_entropy_weighted_loss
        self.entropy_boundary_kernel_size = entropy_boundary_kernel_size
        self.entropy_weighted_alpha = entropy_weighted_alpha
        self.cls_pretrained_epoch = cls_pretrained_epoch
        self.confidence_cls = confidence_cls
        self.ema_alpha_proto = ema_alpha_proto
        if vlc_temperature <= 0:
            raise ValueError("vlc_temperature must be positive")
        self.vlc_temperature = vlc_temperature
        self.compute_consistency_loss = compute_consistency_loss
        self.repulsion_iou_thr = repulsion_iou_thr
        self.repulsion_margin = repulsion_margin
        self.consistency_margin = consistency_margin
        if self.use_cls_tokens:
            assert "loss_cls" in self.weight_dict
        if self.compute_consistency_loss:
            assert "loss_consistency" in self.weight_dict
            assert "loss_repulsion" in self.weight_dict
        logging.info(
            "Using SurgSLOT loss (semantic_tokens=%s)",
            self.use_cls_tokens,
        )

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor, category_batch=None, epoch=None):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        # Per-clip EMA prototype state: [B, D], maintained across frames
        prototype = None
        proto_valid = None  # [B] bool: which batch items have been initialized

        for i, (outs, targets) in enumerate(zip(outs_batch, targets_batch)):
            category = None if category_batch is None else category_batch[i]
            cur_losses = self._forward(outs, targets, num_objects, category, epoch=epoch)

            if self.use_cls_tokens and self.compute_consistency_loss:
                cur_losses, prototype, proto_valid = self._update_prototype(
                    outs, targets, cur_losses, prototype, proto_valid, epoch
                )

            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _update_prototype(self, outs, targets, cur_losses, prototype, proto_valid, epoch):
        """Maintain a per-clip EMA prototype from tracking frames and add consistency
        and repulsion losses.

        Only tracking frames (len(multistep) == 1) contribute.
        Prototype UPDATE gate:  best GT IoU > confidence_cls threshold.
        Consistency LOSS gate:  best GT IoU > confidence_cls AND prototype initialized.
        Repulsion LOSS gate:    best GT IoU < repulsion_iou_thr AND prototype initialized.
          (covers both "object absent" and "segmentation failure" — in both cases the
          CLS token should be dissimilar to the object prototype)
        Both losses are computed BEFORE the EMA update to avoid self-comparison.
        """
        is_tracking_frame = len(outs["multistep_pred_multimasks_high_res"]) == 1
        if not is_tracking_frame:
            return cur_losses, prototype, proto_valid
        if outs.get("is_init_cond_frame", False):
            return cur_losses, prototype, proto_valid

        out_cls_tokens_list = outs["multistep_out_cls_tokens"]
        src_masks = outs["multistep_pred_multimasks_high_res"][-1]  # [B, M, H, W]

        # Normalized cls token for the last (only) prediction step
        raw_vec = out_cls_tokens_list[-1].squeeze(0).float()  # [B, D]
        if epoch is not None and epoch < self.cls_pretrained_epoch:
            raw_vec = raw_vec.detach()
        cur_vec = F.normalize(raw_vec, dim=-1)  # [B, D]

        # GT IoU: measures whether the model truly segmented the correct object
        best_gt_iou = self._compute_best_gt_iou(src_masks, targets.unsqueeze(1).float())  # [B]

        attract_gate   = best_gt_iou > self.confidence_cls
        repulsion_gate = best_gt_iou <= self.repulsion_iou_thr

        # Consistency loss — attract high-IoU frames toward prototype
        loss_consistency = torch.zeros(1, device=cur_vec.device, dtype=cur_vec.dtype).squeeze()
        if self.weight_dict["loss_consistency"] > 0 and prototype is not None and proto_valid.any():
            active = attract_gate & proto_valid
            if active.any():
                sim = (cur_vec[active] * prototype[active]).sum(dim=-1)
                loss_consistency = F.relu(self.consistency_margin - sim).mean()
                cur_losses[CORE_LOSS_KEY] = (
                    cur_losses[CORE_LOSS_KEY] + loss_consistency * self.weight_dict["loss_consistency"]
                )
        cur_losses["loss_consistency"] = loss_consistency.detach()

        # Repulsion loss — push low-IoU frames away from prototype (hinge on cosine sim)
        loss_repulsion = torch.zeros(1, device=cur_vec.device, dtype=cur_vec.dtype).squeeze()
        if self.weight_dict["loss_repulsion"] > 0 and prototype is not None and proto_valid.any():
            active_rep = repulsion_gate & proto_valid
            if active_rep.any():
                sim_neg = (cur_vec[active_rep] * prototype[active_rep]).sum(dim=-1)
                loss_repulsion = F.relu(sim_neg - self.repulsion_margin).mean()
                cur_losses[CORE_LOSS_KEY] = (
                    cur_losses[CORE_LOSS_KEY] + loss_repulsion * self.weight_dict["loss_repulsion"]
                )
        cur_losses["loss_repulsion"] = loss_repulsion.detach()

        # EMA prototype update
        with torch.no_grad():
            cur_vec_d = cur_vec.detach()
            if prototype is None:
                prototype = torch.zeros_like(cur_vec_d)
                proto_valid = torch.zeros(
                    cur_vec_d.shape[0], dtype=torch.bool, device=cur_vec_d.device
                )
            if attract_gate.any():
                new_items = attract_gate & ~proto_valid
                update_items = attract_gate & proto_valid
                if new_items.any():
                    prototype[new_items] = cur_vec_d[new_items]
                    proto_valid[new_items] = True
                if update_items.any():
                    updated = (
                        self.ema_alpha_proto * prototype[update_items]
                        + (1 - self.ema_alpha_proto) * cur_vec_d[update_items]
                    )
                    prototype[update_items] = F.normalize(updated, dim=-1)

        return cur_losses, prototype, proto_valid

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects,
                 target_category=None, epoch=None):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]
        out_cls_tokens = outputs.get(
            "multistep_out_cls_tokens", [None] * len(src_masks_list)
        )
        # all_category_tokens = outputs.get("all_category_tokens", None)
        all_category_text_features = outputs.get("text_features", None)

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        is_init_cond_frames = outputs.get("is_init_cond_frame", False)
        # accumulate the loss over prediction steps
        _zero = torch.tensor(0.0, dtype=torch.float32, device=targets.device)
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0, "loss_cls": 0, "loss_consistency": _zero, "loss_repulsion": _zero}
        for i, (src_masks, ious, object_score_logits) in enumerate(zip(
            src_masks_list, ious_list, object_score_logits_list
        )):
            this_out_cls_tokens = out_cls_tokens[i] if self.use_cls_tokens else None

            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits, this_out_cls_tokens,
                all_category_text_features=all_category_text_features, target_category=target_category,
                is_init_cond_frames=is_init_cond_frames, epoch=epoch
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits,
            out_cls_tokens=None, target_category=None, all_category_text_features=None, is_init_cond_frames=False,
            epoch=None
    ):
        target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        if self.use_entropy_weighted_loss:
            boundary_entropy_weights = compute_boundary_entropy_weights(src_masks, target_masks,
                                                                        kernel_size=self.entropy_boundary_kernel_size,
                                                                        alpha=self.entropy_weighted_alpha)
            loss_multimask = sigmoid_focal_loss(
                src_masks,
                target_masks,
                num_objects,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                loss_on_multimask=True,
                pixel_weights=boundary_entropy_weights,
            )
        else:
            loss_multimask = sigmoid_focal_loss(
                src_masks,
                target_masks,
                num_objects,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                loss_on_multimask=True,
            )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )
        # Skip classification loss when target_category is None.
        # Conditional frames also skip CLS loss because classification from a single image
        # is unstable and its feature distribution differs from memory-conditioned features.
        # A mask-input CLS token comes directly from raw_cls and carries no useful information.
        if is_init_cond_frames or not self.use_cls_tokens or target_category is None:
            loss_cls = torch.tensor(0.0, dtype=loss_multimask.dtype, device=loss_multimask.device)
        else:
            # Stage one: stop gradients from propagating back into the tracker.
            if epoch is not None and epoch < self.cls_pretrained_epoch:
                out_cls_tokens = out_cls_tokens.detach()

            # Gate cls loss on GT IoU: only learn from frames where the model truly
            # segmented the correct object (avoids reinforcing wrong predictions).
            best_gt_iou = self._compute_best_gt_iou(src_masks, target_masks[:, :1])  # [B]
            confident_mask = (best_gt_iou > self.confidence_cls).float()  # [B]

            out_cls_tokens = F.normalize(out_cls_tokens, dim=-1)
            out_cls_tokens = out_cls_tokens.squeeze(0)  # [1, B, D] -> [B, D]
            all_category_text_features = F.normalize(all_category_text_features, dim=-1)
            pred_logits = self.get_sim_logits(
                all_category_text_features,
                out_cls_tokens,
                temperature=self.vlc_temperature,
            )
            # Cross-entropy applies softmax to pred_logits internally.
            gt_label = target_category
            # Return zero loss when every ground-truth label is -1.
            if (gt_label == -1).all():
                loss_cls = torch.tensor(0.0, dtype=loss_multimask.dtype, device=loss_multimask.device)
            else:
                # Account for both ignore_index and the confidence gate.
                valid_mask = (gt_label != -1).float()  # [B]
                final_mask = valid_mask * confident_mask  # [B]
                per_sample_loss = F.cross_entropy(
                    pred_logits, gt_label, ignore_index=-1, reduction='none'
                )  # [B]
                masked_loss = per_sample_loss * final_mask
                num_valid = final_mask.sum().clamp(min=1.0)
                loss_cls = masked_loss.sum() / num_valid
                pass

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class
        losses["loss_cls"] += loss_cls

    def _compute_best_gt_iou(self, src_masks, target_masks):
        """Compute best-over-candidates GT IoU for each batch item. When GT is empty, the iou will be always 0

        Args:
            src_masks:    [B, M, H, W] predicted mask logits
            target_masks: [B, 1, H, W] binary GT masks
        Returns:
            best_gt_iou: [B] float tensor in [0, 1]
        """
        pred = src_masks.flatten(2) > 0        # [B, M, H*W]
        gt = target_masks.flatten(2) > 0       # [B, 1, H*W]
        area_i = (pred & gt).float().sum(-1)   # [B, M]
        area_u = (pred | gt).float().sum(-1)   # [B, M]
        gt_ious = area_i / area_u.clamp(min=1.0)  # [B, M]
        best_gt_iou, _ = gt_ious.max(dim=1)   # [B]
        return best_gt_iou

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss

    def get_sim_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        temperature: float = 0.01,
    ):
        return image_features @ text_features.T / temperature
