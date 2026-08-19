# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
ClsMemoryAttention: memory attention where cls tokens are excluded from
self-attention but participate in cross-attention.

Ported from SurgicalSAM2.1/sam2/modeling/cls_memory_attention.py.
Key change vs TransformerEncoderCrossAttention: the _forward_sa of each layer
completely skips cls tokens (they are passed through unchanged), whereas the
original code only excludes cls from RoPE but still includes them in SA.
"""

from typing import Optional

import torch
from torch import nn, Tensor

from sam3.model.act_ckpt_utils import activation_ckpt_wrapper
from sam3.model.decoder import TransformerDecoderLayerv2, TransformerEncoderCrossAttention
from sam3.sam.transformer import RoPEAttention


class ClsMemoryAttentionLayer(TransformerDecoderLayerv2):
    """
    TransformerDecoderLayerv2 variant that completely excludes cls tokens from
    self-attention. Cls tokens are kept unchanged during SA and only participate
    in cross-attention (memory attention).
    """

    def _forward_sa(self, tgt, query_pos, num_q_exclude_rope_s: int = 0, num_cls_tokens: int = 0):
        if num_cls_tokens > 0:
            # tgt is batch-first [B, L, D]; cls occupies the first num_cls_tokens positions.
            cls_part = tgt[:, :num_cls_tokens, :]          # [B, C, D] — passed through unchanged
            frame_part = tgt[:, num_cls_tokens:, :]        # [B, N, D]
            frame_pos = query_pos[:, num_cls_tokens:, :]   # [B, N, D]

            frame2 = self.norm1(frame_part)
            q = k = frame2 + frame_pos if self.pos_enc_at_attn else frame2
            # RoPE is applied to all frame tokens (no exclusion needed — cls is already split out)
            frame2 = self.self_attn(q, k, v=frame2)
            frame_part = frame_part + self.dropout1(frame2)
            return torch.cat([cls_part, frame_part], dim=1)

        # Standard path: no cls tokens, delegate to parent
        return super()._forward_sa(tgt, query_pos, num_q_exclude_rope_s)

    def forward_pre(
        self,
        tgt,
        memory,
        dac: bool = False,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        attn_bias=None,
        num_k_exclude_rope: int = 0,
        num_cls_tokens: int = 0,
        num_q_exclude_rope_s: int = 0,
    ):
        assert dac is False
        assert tgt_mask is None
        assert memory_mask is None
        assert tgt_key_padding_mask is None
        assert memory_key_padding_mask is None
        assert attn_bias is None

        if self.cross_attention_first:
            # CA first: cls token attends to memory, then SA (skipping cls)
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope, num_q_exclude_rope_s)
            tgt = self._forward_sa(tgt, query_pos, num_cls_tokens=num_cls_tokens)
        else:
            # SA first (skipping cls), then CA: cls token attends to memory
            tgt = self._forward_sa(tgt, query_pos, num_cls_tokens=num_cls_tokens)
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope, num_q_exclude_rope_s)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, *args, **kwds):
        if self.pre_norm:
            return self.forward_pre(*args, **kwds)
        raise NotImplementedError


class ClsMemoryAttention(TransformerEncoderCrossAttention):
    """
    Memory attention encoder where cls tokens skip self-attention entirely.

    Drop-in replacement for TransformerEncoderCrossAttention when layers are
    ClsMemoryAttentionLayer instances. The only behavioural difference is that
    cls tokens (prepended to the sequence) are excluded from self-attention in
    each layer via `num_cls_tokens=1`, while cross-attention is unchanged.
    """

    def forward(
        self,
        src,
        prompt,
        src_mask: Optional[Tensor] = None,
        prompt_mask: Optional[Tensor] = None,
        src_key_padding_mask=None,
        prompt_key_padding_mask=None,
        src_pos: Optional[Tensor] = None,
        prompt_pos: Optional[Tensor] = None,
        feat_sizes=None,
        num_obj_ptr_tokens: int = 0,
        cls_tokens: Optional[Tensor] = None,
    ):
        if isinstance(src, list):
            assert isinstance(src_key_padding_mask, list) and isinstance(src_pos, list)
            assert len(src) == len(src_key_padding_mask) == len(src_pos) == 1
            src, src_key_padding_mask, src_pos = (
                src[0],
                src_key_padding_mask[0],
                src_pos[0],
            )

        assert src.shape[1] == prompt.shape[1], (
            "Batch size must be the same for src and prompt"
        )

        output = src

        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        # Prepend cls_token before batch_first conversion (seq-first space)
        if cls_tokens is not None:
            output = torch.cat([cls_tokens, output], dim=0)   # (HW+1, B, D)
            zero_pos = torch.zeros_like(cls_tokens)
            src_pos = torch.cat([zero_pos, src_pos], dim=0)   # (HW+1, B, D)

        if self.batch_first:
            output = output.transpose(0, 1)      # (B, HW+1, D)
            src_pos = src_pos.transpose(0, 1)
            prompt = prompt.transpose(0, 1)
            prompt_pos = prompt_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds["num_k_exclude_rope"] = num_obj_ptr_tokens
            if cls_tokens is not None:
                kwds["num_cls_tokens"] = 1       # exclude cls from self-attention
                kwds["num_q_exclude_rope_s"] = 1  # exclude cls from RoPE in cross-attention

            output = activation_ckpt_wrapper(layer)(
                tgt=output,
                memory=prompt,
                tgt_mask=src_mask,
                memory_mask=prompt_mask,
                tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=prompt_key_padding_mask,
                pos=prompt_pos,
                query_pos=src_pos,
                dac=False,
                attn_bias=None,
                act_ckpt_enable=self.training and self.use_act_checkpoint,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)

        # Extract cls_token from output
        memory_cls_tokens = None
        if cls_tokens is not None:
            memory_cls_tokens = normed_output[:1, :, :]   # (1, B, D)
            normed_output = normed_output[1:, :, :]        # (HW, B, D)
            src_pos = src_pos[1:, :, :]

        return {
            "memory": normed_output,
            "pos_embed": src_pos,
            "padding_mask": src_key_padding_mask,
            "memory_cls_tokens": memory_cls_tokens,
        }