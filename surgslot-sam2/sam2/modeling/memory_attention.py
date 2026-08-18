# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos, num_q_exclude_rope_s=0, num_k_exclude_rope_s=0):
        kwds = {}
        if num_q_exclude_rope_s > 0:
            assert isinstance(self.self_attn, RoPEAttention)
            kwds["num_q_exclude_rope_s"] = num_q_exclude_rope_s
            kwds["num_k_exclude_rope_s"] = num_k_exclude_rope_s
            assert num_q_exclude_rope_s == num_k_exclude_rope_s, "num_q_exclude_rope_s and num_k_exclude_rope_s must be equal"
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2, **kwds)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope_e=0, num_q_exclude_rope_s=0):
        kwds = {}
        if num_k_exclude_rope_e > 0 or num_q_exclude_rope_s > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {
                "num_k_exclude_rope_e": num_k_exclude_rope_e,
                "num_q_exclude_rope_s": num_q_exclude_rope_s,
            }

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_q_exclude_rope_s: int = 0,  # Number of tokens to exclude from RoPE in queries
        num_k_exclude_rope_s: int = 0,  # Number of tokens to exclude from RoPE in keys, starting from the first token
        num_k_exclude_rope_e: int = 0,  # Number of tokens to exclude from RoPE in keys, starting from the end token
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(
            tgt, query_pos,
            num_q_exclude_rope_s=num_q_exclude_rope_s, num_k_exclude_rope_s=num_k_exclude_rope_s
        )
        tgt = self._forward_ca(
            tgt, memory, query_pos, pos,
            num_q_exclude_rope_s=num_q_exclude_rope_s, num_k_exclude_rope_e=num_k_exclude_rope_e
        )
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
        cls_tokens: Optional[torch.Tensor] = None,  # cls token for memory
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr

        memory_cls_tokens = None
        if cls_tokens is not None:
            # Prepend the semantic query to the current-frame tokens.
            output = torch.cat([cls_tokens, curr], dim=0)
            zero_pos = torch.zeros_like(curr_pos[:1, :, :])
            curr_pos = torch.cat([zero_pos, curr_pos], dim=0)

        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope_e": num_obj_ptr_tokens}
                if cls_tokens is not None:
                    kwds["num_q_exclude_rope_s"] = 1
                    kwds["num_k_exclude_rope_s"] = 1

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        if cls_tokens is not None:
            # Remove the cls token from the output
            memory_cls_tokens = normed_output[:1, :, :]
            normed_output = normed_output[1:, :, :]

        # Always return a pair so SAM2 and SurgSLOT share one stable interface.
        return normed_output, memory_cls_tokens
