# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor, nn
from torch.nn import Parameter
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.quant_noise import quant_noise


@with_incremental_state
class LongformerMultiheadAttention(nn.Module):
    """Multi-headed attention with Longformer extension.
    
    references:
    (1) https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb
    (2) See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        attention_window=512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)

        # Separate projections for the global attention
        self.k_proj_global = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj_global = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj_global = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)


        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attention_window_size = attention_window // 2

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))

            nn.init.xavier_uniform_(self.k_proj_global.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj_global.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj_global.weight, gain=1 / math.sqrt(2))

        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

            nn.init.xavier_uniform_(self.k_proj_global.weight)
            nn.init.xavier_uniform_(self.v_proj_global.weight)
            nn.init.xavier_uniform_(self.q_proj_global.weight)


        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert not before_softmax 
        assert not static_kv
        assert incremental_state is None

        saved_state = None

        if attn_mask is not None:
            key_padding_mask = attn_mask < 0
            extra_attention_mask = attn_mask > 0
            remove_from_windowed_attention_mask = attn_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                # To support the case of variable number of global attention in the rows of a batch,
                # we use the following three selection masks to select global attention embeddings
                # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
                # 1) selecting embeddings that correspond to global attention
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(
                    0, max_num_extra_indices_per_batch, device=num_extra_indices_per_batch.device
                )
                # mask indicating which values are actually going to be padding
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                # 2) location of the non-padding values in the selected global attention
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                # 3) location of the padding values in the selected global attention
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None


        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q *= self.scaling

        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        # attn_weights = (batch_size, seqlen, num_heads, window*2+1)
        attn_weights = self._sliding_chunks_matmul_qk(q, k, self.one_sided_attention_window_size)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (batch_size x seqlen) to (batch_size x seqlen x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(
                dim=-1
            )
            # cast to fp32/fp16 then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(
                remove_from_windowed_attention_mask, -10000.0
            )
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            d_mask = self._sliding_chunks_matmul_qk(ones, float_mask, self.one_sided_attention_window_size)
            attn_weights += d_mask
        assert list(attn_weights.size()) == [
            bsz,
            tgt_len,
            self.num_heads,
            self.one_sided_attention_window_size * 2 + 1,
        ]

        # the extra attention
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            # (batch_size, seqlen, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum("blhd,bshd->blhs", (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000.0
            # concat to attn_weights
            # (batch_size, seqlen, num_heads, extra attention count + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)# the extra attention

        attn_weights_fp32 = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        attn_weights = attn_weights_fp32.type_as(attn_weights)

        src_len = tgt_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len


        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights = torch.masked_fill(attn_weights, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        v = v.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        attn = None

        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            # use `matmul` because `einsum` crashes sometimes with fp16
            # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn = torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2)).transpose(1, 2)
            attn_probs = attn_probs.narrow(
                -1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch
            ).contiguous()
        if attn is None:
            attn = self._sliding_chunks_matmul_pv(attn_probs, v, self.one_sided_attention_window_size)
        else:
            attn += self._sliding_chunks_matmul_pv(attn_probs, v, self.one_sided_attention_window_size)

        assert attn.size() == (bsz, tgt_len, self.num_heads, self.head_dim), "Unexpected size"
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).contiguous()


        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor.
        # TODO: remove the redundant computation
        if extra_attention_mask is not None:
            selected_query = query.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
            selected_query[selection_padding_mask_nonzeros[::-1]] = query[
                extra_attention_mask_nonzeros[::-1]
            ]

            q = self.q_proj_global(selected_query)
            k = self.k_proj_global(query)
            v = self.v_proj_global(query)
            q *= self.scaling

            q = (
                q.contiguous()
                .view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )  # (batch_size * self.num_heads, max_num_extra_indices_per_batch, head_dim)
            k = (
                k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            )  # batch_size * self.num_heads, seqlen, head_dim)
            v = (
                v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            )  # batch_size * self.num_heads, seqlen, head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, tgt_len]

            attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, tgt_len)
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -10000.0,)
            attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, tgt_len)
            attn_weights_float = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            )  # use fp32 for numerical stability
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [
                bsz * self.num_heads,
                max_num_extra_indices_per_batch,
                self.head_dim,
            ]

            selected_attn_4d = selected_attn.view(
                bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim
            )
            nonzero_selected_attn = selected_attn_4d[
                selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]
            ]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(
                len(selection_padding_mask_nonzeros[0]), -1
            )

        attn = self.out_proj(attn)
        final_attn_weights: Optional[Tensor] = None
        if need_weights:
            if extra_attention_mask is not None:
                final_attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, tgt_len).transpose(0,1)
              
            else:
                final_attn_weights = attn_weights.permute(0, 2, 1, 3).transpose(0,1)

            if not need_head_weights:
                # average attention weights over heads
                final_attn_weights = final_attn_weights.mean(dim=0)


        return attn, final_attn_weights
    
    
    @staticmethod
    def _skew(x, direction):
        """Convert diagonals into columns (or columns into diagonals depending on `direction`"""
        x_padded = F.pad(x, direction)  # padding value is not important because it will be overwritten
        x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
        return x_padded

    @staticmethod
    def _skew2(x):
        """shift every row 1 step to right converting columns into diagonals"""
        # X = B x C x M x L
        B, C, M, L = x.size()
        x = F.pad(x, (0, M + 1))  # B x C x M x (L+M+1). Padding value is not important because it'll be overwritten
        x = x.view(B, C, -1)  # B x C x ML+MM+M
        x = x[:, :, :-M]  # B x C x ML+MM
        x = x.view(B, C, M, M + L)  # B x C, M x L+M
        x = x[:, :, :, :-1]
        return x

    @staticmethod
    def _chunk(x, w):
        """convert into overlapping chunkings. Chunk size = 2w, overlap size = w"""

        # non-overlapping chunks of size = 2w
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

        # use `as_strided` to make the chunks overlap with an overlap size = w
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)
    
    def _mask_invalid_locations(self, input_tensor, w) -> torch.Tensor:
        affected_seqlen = w
        beginning_mask_2d = input_tensor.new_ones(w, w + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seqlen, :, : w + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
        ending_input = input_tensor[:, -affected_seqlen:, :, -(w + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8


    def _sliding_chunks_matmul_qk(self, q: torch.Tensor, k: torch.Tensor, w: int):
        """Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
        with an overlap of size w"""
        batch_size, seqlen, num_heads, head_dim = q.size()
        assert seqlen % (w * 2) == 0, f"Sequence length should be multiple of {w * 2}. Given {seqlen}"
        assert q.size() == k.size()

        chunks_count = seqlen // w - 1

        # group batch_size and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
        q = q.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)
        k = k.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)

        chunk_q = self._chunk(q, w)
        chunk_k = self._chunk(k, w)

        # matrix multipication
        # bcxd: batch_size * num_heads x chunks x 2w x head_dim
        # bcyd: batch_size * num_heads x chunks x 2w x head_dim
        # bcxy: batch_size * num_heads x chunks x 2w x 2w
        chunk_attn = torch.einsum("bcxd,bcyd->bcxy", (chunk_q, chunk_k))  # multiply

        # convert diagonals into columns
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1))

        # allocate space for the overall attention matrix where the chunks are compined. The last dimension
        # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
        # w previous words). The following column is attention score from each word to itself, then
        # followed by w columns for the upper triangle.

        diagonal_attn = diagonal_chunk_attn.new_empty((batch_size * num_heads, chunks_count + 1, w, w * 2 + 1))

        # copy parts from diagonal_chunk_attn into the compined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, : w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, : w + 1]
        # - copying the lower triangle
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1) : -1, w + 1 :]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, : w - 1, 1 - w :]

        # separate batch_size and num_heads dimensions again
        diagonal_attn = diagonal_attn.view(batch_size, num_heads, seqlen, 2 * w + 1).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attn, w)
        return diagonal_attn


    def _sliding_chunks_matmul_pv(self, prob: torch.Tensor, v: torch.Tensor, w: int):
        """Same as _sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
        format from _sliding_chunks_matmul_qk"""
        batch_size, seqlen, num_heads, head_dim = v.size()
        assert seqlen % (w * 2) == 0
        assert prob.size()[:3] == v.size()[:3]
        assert prob.size(3) == 2 * w + 1
        chunks_count = seqlen // w - 1
        # group batch_size and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
        chunk_prob = prob.transpose(1, 2).reshape(batch_size * num_heads, seqlen // w, w, 2 * w + 1)

        # group batch_size and num_heads dimensions into one
        v = v.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)

        # pad seqlen with w at the beginning of the sequence and another w at the end
        padded_v = F.pad(v, (0, 0, w, w), value=-1)

        # chunk padded_v into chunks of size 3w and an overlap of size w
        chunk_v_size = (batch_size * num_heads, chunks_count + 1, 3 * w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

        skewed_prob = self._skew2(chunk_prob)

        context = torch.einsum("bcwd,bcdh->bcwh", (skewed_prob, chunk_v))
        return context.view(batch_size, num_heads, seqlen, head_dim).transpose(1, 2) 
 
    


    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
