# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch MiniCPM model."""
import math
import warnings
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, \
    SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_minicpm import MiniCPMConfig
import re
import json

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    pass

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

# import aimet_torch.elementwise_ops as op

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MiniCPMConfig"
# save_activation_dynamic_range = True
# save_min_max = True
# # use_negative_100 = True # NOTE: pro-1b在modeling_attn_mask_utils.py里面调整
# use_qualcomm = True
# static_quant = False

save_activation_dynamic_range = False
save_min_max = False
# use_negative_100 = True # NOTE: pro-1b在modeling_attn_mask_utils.py里面调整
use_qualcomm = True
static_quant = True

max_int = 32767
dynamic_range_dict = {}
symmetric=False

dynamic_range_dict_path = '/home/workspace/code/git/AutoGPTQ_mlm/auto_gptq/modeling/minicpm_new/max_min_values.json'

with open(dynamic_range_dict_path, 'r') as file:
    dynamic_range_dict_load = json.load(file)

# def asymmetric_fake_quant(tensor, min_max_list):
#     min_val = torch.tensor(min_max_list[0], dtype=torch.float32, device=tensor.device)
#     max_val = torch.tensor(min_max_list[1], dtype=torch.float32, device=tensor.device)
#     max_int_t = torch.tensor(max_int, dtype=torch.float32, device=tensor.device)
    
#     # 计算s和z
#     s = (max_val - min_val) / (2 * max_int_t + 1)
#     # z = torch.clamp(max_int_t - torch.round(max_val / s), -max_int -1 , max_int)
#     z = max_int_t - torch.round(max_val / s)
#     # 执行量化和反量化操作
#     tensor_quantized = torch.clamp(torch.round(tensor / s + z), -max_int - 1, max_int)
#     tensor_dequantized = (tensor_quantized - z) * s
#     # 计算均方误差
#     mse = torch.mean((tensor - tensor_dequantized) ** 2)

#     threshold=20
#     if mse > threshold:
#         print(f"High MSE: {mse.item()}, min_max_list: {min_max_list}")
#         print("tensor: ", tensor)
#         print("tensor_dequantized: ", tensor_dequantized)
#         exit(0)
#     if torch.isnan(mse).any():
#         print(f"Nan MSE: {mse.item()}, min_max_list: {min_max_list}")
#         exit(0)
#     return tensor_dequantized

# def asymmetric_fake_quant(tensor, min_max_list, name=None):
#     # max_abs_value = max(abs(x) for x in min_max_list)
#     # min_1 = - max_abs_value
#     # max_1 = max_abs_value
#     fix_ratio = 1.0
#     if min_max_list[0]<0:
#         min_1=min_max_list[0]*fix_ratio
#     else:
#         min_1=min_max_list[0]
#     if min_max_list[1]>0:
#         max_1=min_max_list[1]*fix_ratio
#     else:
#         max_1=min_max_list[1]
#     min_val = torch.tensor(min_1, dtype=torch.float32, device=tensor.device)
#     max_val = torch.tensor(max_1, dtype=torch.float32, device=tensor.device)
#     max_int_t = torch.tensor(max_int, dtype=torch.float32, device=tensor.device)
    
#     # 计算s和z
#     s = (max_val - min_val) / (2 * max_int_t + 1)
#     # z = torch.clamp(max_int_t - torch.round(max_val / s), -max_int -1 , max_int)
#     z = max_int_t - torch.round(max_val / s).clamp_(-max_int-1, max_int)
#     # 执行量化和反量化操作
#     tensor_quantized = torch.clamp(torch.round(tensor / s + z), -max_int - 1, max_int)
#     tensor_dequantized = (tensor_quantized - z) * s
#     # 计算均方误差
#     # mse = torch.mean((tensor - tensor_dequantized) ** 2)

#     # threshold=10
#     # if mse > threshold:
#     #     print(name, f" High MSE: {mse.item()}, min_max_list: {min_max_list}")
#     #     # print("tensor: ", tensor)
#     #     # print("tensor_dequantized: ", tensor_dequantized)
#     #     # exit(0)
#     # if torch.isnan(mse).any():
#     #     print(name, f" Nan MSE: {mse.item()}, min_max_list: {min_max_list}")
#     #     exit(0)
#     return tensor_dequantized

def asymmetric_fake_quant(tensor, min_max_list, name=None):
    dtype = tensor.dtype
    tensor = tensor.to(torch.float32)
    min_val = torch.tensor(min_max_list[0], dtype=torch.float32, device=tensor.device)
    max_val = torch.tensor(min_max_list[1], dtype=torch.float32, device=tensor.device)
    max_int = 2**16-1
    max_int_t = torch.tensor(max_int, dtype=torch.float32, device=tensor.device)
    
    s = (max_val - min_val) / max_int_t
    s = s.to(torch.float32)
    z = (-torch.round(min_val / s)).clamp_(0, max_int)
    z = z.to(torch.float32)
    tensor_quantized = torch.clamp(torch.round(tensor/s)+z, 0, max_int)
    tensor_dequantized = (tensor_quantized - z) * s
    mse = torch.mean((tensor - tensor_dequantized) ** 2)

    threshold=0.01
    if mse > threshold:
        print(f"High MSE: {mse.item()}, min_max_list: {min_max_list}, name: {name}")
    if torch.isnan(mse).any():
        print(f"Nan MSE: {mse.item()}, min_max_list: {min_max_list}, name: {name}")
    return tensor_dequantized.to(dtype)

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.minicpm.modeling_minicpm._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.minicpm.modeling_minicpm._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.minicpm.modeling_minicpm.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


# @torch.jit.script  # type: ignore
def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniCPMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(MiniCPMRMSNorm)


class MiniCPMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            # seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MiniCPMLongRoPE(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, short_factor=None, long_factor=None, original_max_position_embeddings=None):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = (max_position_embeddings /
                 self.original_max_position_embeddings)
        self.scaling_factor = math.sqrt(
                1 + math.log(scale) /
                math.log(self.original_max_position_embeddings))
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=device)
        
        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype)
        )
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype) * self.scaling_factor, persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype) * self.scaling_factor, persistent=False)


class MiniCPMLinearScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MiniCPMDynamicNTKScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    # print("pos: ",position_ids.shape)
    # print("cos: ",cos.shape) # 这里不太对
    # print("sin: ",sin.shape)
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


def apply_rotary_pos_emb_qualcomm(q, k, cos, sin, position_ids, layer_idx=None, unsqueeze_dim=1):
    '''
    Based on FacebookResearch's llama, provided by Carl
    '''
    # rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
    # rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2
    orig_dtype = k.dtype
    rope_real = cos[position_ids].unsqueeze(unsqueeze_dim)[..., : cos[position_ids].shape[-1] // 2] # shape有问题
    rope_im = sin[position_ids].unsqueeze(unsqueeze_dim)[..., : sin[position_ids].shape[-1] // 2]
    if save_activation_dynamic_range:
        if save_min_max:
            vmin, vmax = rope_real.min().item(), rope_real.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.rope_real'] = [vmin, vmax]
            vmin, vmax = rope_im.min().item(), rope_im.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.rope_im'] = [vmin, vmax]
        else:
            vmax = rope_real.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.rope_real'] = vmax
            vmax = rope_im.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.rope_im'] = vmax
    
    if static_quant:
        if not symmetric:
            rope_real = asymmetric_fake_quant(rope_real, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.rope_real'], f'model.layers.{layer_idx}.self_attn.rope_real')
            rope_im = asymmetric_fake_quant(rope_im, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.rope_im'],f'model.layers.{layer_idx}.self_attn.rope_im')
        # TODO: if symmetric:


    # TODO: Why HF uses different coordinates from the paper
    # x_real = x[:,:,:,:x.shape[-1]//2] # extract first half elements
    # x_im = x[:,:,:,x.shape[-1]//2:] # extract second half elements
    q_real = q[..., : q.shape[-1] // 2]
    q_im = q[..., q.shape[-1] // 2 :]
    k_real = k[..., : k.shape[-1] // 2]
    k_im = k[..., k.shape[-1] // 2 :]
    if save_activation_dynamic_range:
        if save_min_max:
            vmin, vmax = q_real.min().item(), q_real.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_real'] = [vmin, vmax]
            vmin, vmax = q_im.min().item(), q_im.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_im'] = [vmin, vmax]
            vmin, vmax = k_real.min().item(), k_real.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_real'] = [vmin, vmax]
            vmin, vmax = k_im.min().item(), k_im.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_im'] = [vmin, vmax]
        else:
            vmax = q_real.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_real'] = vmax
            vmax = q_im.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_im'] = vmax
            vmax = k_real.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_real'] = vmax
            vmax = k_im.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_im'] = vmax

    if static_quant:
        if not symmetric:
            q_real = asymmetric_fake_quant(q_real, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.q_real'],f'model.layers.{layer_idx}.self_attn.q_real')
            q_im = asymmetric_fake_quant(q_im, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.q_im'],f'model.layers.{layer_idx}.self_attn.q_im')
            k_real = asymmetric_fake_quant(k_real, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.k_real'],f'model.layers.{layer_idx}.self_attn.k_real')
            k_im = asymmetric_fake_quant(k_im, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.k_im'], f'model.layers.{layer_idx}.self_attn.k_im')

    # x_prod_real = x_real*rope_real - x_im * rope_im
    # x_prod_im = x_real*rope_im + x_im*rope_real
    if static_quant: # no static_quant
        if not symmetric:
            q_real_mul_rope_real = asymmetric_fake_quant(q_real*rope_real, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.q_real*rope_real'],f'model.layers.{layer_idx}.self_attn.q_real*rope_real')
            q_im_mul_rope_im = asymmetric_fake_quant(q_im*rope_im, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.q_im*rope_im'],f'model.layers.{layer_idx}.self_attn.q_im*rope_im')
            q_real_mul_rope_im = asymmetric_fake_quant(q_real*rope_im, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.q_real*rope_im'],f'model.layers.{layer_idx}.self_attn.q_real*rope_im')
            q_im_mul_rope_real = asymmetric_fake_quant(q_im*rope_real, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.q_im*rope_real'],f'model.layers.{layer_idx}.self_attn.q_im*rope_real')
            k_real_mul_rope_real = asymmetric_fake_quant(k_real*rope_real, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.k_real*rope_real'],f'model.layers.{layer_idx}.self_attn.k_real*rope_real')
            k_im_mul_rope_im = asymmetric_fake_quant(k_im*rope_im, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.k_im*rope_im'],f'model.layers.{layer_idx}.self_attn.k_im*rope_im')
            k_real_mul_rope_im = asymmetric_fake_quant(k_real*rope_im, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.k_real*rope_im'],f'model.layers.{layer_idx}.self_attn.k_real*rope_im')
            k_im_mul_rope_real = asymmetric_fake_quant(k_im*rope_real, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.k_im*rope_real'],f'model.layers.{layer_idx}.self_attn.k_im*rope_real')
            q_prod_real = q_real_mul_rope_real - q_im_mul_rope_im 
            q_prod_im = q_real_mul_rope_im + q_im_mul_rope_real
            k_prod_real = k_real_mul_rope_real - k_im_mul_rope_im
            k_prod_im = k_real_mul_rope_im + k_im_mul_rope_real
            q_prod_real = asymmetric_fake_quant(q_prod_real, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.q_prod_real'],f'model.layers.{layer_idx}.self_attn.q_prod_real')
            q_prod_im = asymmetric_fake_quant(q_prod_im, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.q_prod_im'],f'model.layers.{layer_idx}.self_attn.q_prod_im')
            k_prod_real = asymmetric_fake_quant(k_prod_real, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.k_prod_real'],f'model.layers.{layer_idx}.self_attn.k_prod_real')
            k_prod_im = asymmetric_fake_quant(k_prod_im, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.k_prod_im'],f'model.layers.{layer_idx}.self_attn.k_prod_im')
    else:
        q_prod_real = q_real*rope_real - q_im * rope_im
        q_prod_im = q_real*rope_im + q_im*rope_real
        k_prod_real = k_real*rope_real - k_im * rope_im
        k_prod_im = k_real*rope_im + k_im*rope_real
    if save_activation_dynamic_range:
        if save_min_max:
            vmin, vmax = (q_real * rope_real).min().item(), (q_real * rope_real).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_real*rope_real'] = [vmin, vmax]
            vmin, vmax = (q_im * rope_im).min().item(), (q_im * rope_im).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_im*rope_im'] = [vmin, vmax]
            vmin, vmax = (q_real * rope_im).min().item(), (q_real * rope_im).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_real*rope_im'] = [vmin, vmax]
            vmin, vmax = (q_im * rope_real).min().item(), (q_im * rope_real).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_im*rope_real'] = [vmin, vmax]
            vmin, vmax = (k_real * rope_real).min().item(), (k_real * rope_real).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_real*rope_real'] = [vmin, vmax]
            vmin, vmax = (k_im * rope_im).min().item(), (k_im * rope_im).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_im*rope_im'] = [vmin, vmax]
            vmin, vmax = (k_real * rope_im).min().item(), (k_real * rope_im).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_real*rope_im'] = [vmin, vmax]
            vmin, vmax = (k_im * rope_real).min().item(), (k_im * rope_real).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_im*rope_real'] = [vmin, vmax]
            vmin, vmax = q_prod_real.min().item(), q_prod_real.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_prod_real'] = [vmin, vmax]
            vmin, vmax = q_prod_im.min().item(), q_prod_im.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_prod_im'] = [vmin, vmax]
            vmin, vmax = k_prod_real.min().item(), k_prod_real.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_prod_real'] = [vmin, vmax]
            vmin, vmax = k_prod_im.min().item(), k_prod_im.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_prod_im'] = [vmin, vmax]
        else:
            vmax = (q_real*rope_real).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_real*rope_real'] = vmax
            vmax = (q_im * rope_im).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_im*rope_im'] = vmax
            vmax = (q_real*rope_im).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_real*rope_im'] = vmax
            vmax = (q_im*rope_real).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_im*rope_real'] = vmax
            vmax = (k_real*rope_real).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_real*rope_real'] = vmax
            vmax = (k_im * rope_im).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_im*rope_im'] = vmax
            vmax = (k_real*rope_im).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_real*rope_im'] = vmax
            vmax = (k_im*rope_real).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_im*rope_real'] = vmax
            vmax = q_prod_real.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_prod_real'] = vmax
            vmax = q_prod_im.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.q_prod_im'] = vmax
            vmax = k_prod_real.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_prod_real'] = vmax
            vmax = k_prod_im.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.k_prod_im'] = vmax

            
    # TODO: HF need to uses different interleaving
    # x = torch.cat((x_prod_real,x_prod_im),dim=3).view(*x.shape)
    q_embed = torch.cat((q_prod_real,q_prod_im),dim=3).view(*q.shape)
    k_embed = torch.cat((k_prod_real,k_prod_im),dim=3).view(*k.shape)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


def apply_rotary_pos_emb_qualcomm_single(name, x, cos, sin, position_ids, layer_idx=None, idx=None, unsqueeze_dim=1):
    '''
    Based on FacebookResearch's llama, provided by Carl
    '''
    # rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
    # rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2
    orig_dtype = x.dtype
    rope_real = cos[position_ids].unsqueeze(unsqueeze_dim)[..., : cos[position_ids].shape[-1] // 2] # shape有问题
    rope_im = sin[position_ids].unsqueeze(unsqueeze_dim)[..., : sin[position_ids].shape[-1] // 2]
    if save_activation_dynamic_range:
        if save_min_max:
            vmin, vmax = rope_real.min().item(), rope_real.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.rope_real.{idx}'] = [vmin, vmax]
            vmin, vmax = rope_im.min().item(), rope_im.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.rope_im.{idx}'] = [vmin, vmax]
        else:
            vmax = rope_real.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.rope_real.{idx}'] = vmax
            vmax = rope_im.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.rope_im.{idx}'] = vmax

    if static_quant:
        if not symmetric:
            rope_real = asymmetric_fake_quant(rope_real, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.rope_real.{idx}'],f'model.layers.{layer_idx}.self_attn.rope_real.{idx}')
            rope_im = asymmetric_fake_quant(rope_im, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.rope_im.{idx}'],f'model.layers.{layer_idx}.self_attn.rope_im.{idx}')
        # TODO: if symmetric:

    # TODO: Why HF uses different coordinates from the paper
    # x_real = x[:,:,:,:x.shape[-1]//2] # extract first half elements
    # x_im = x[:,:,:,x.shape[-1]//2:] # extract second half elements
    x_real = x[..., : x.shape[-1] // 2]
    x_im = x[..., x.shape[-1] // 2 :]
    if save_activation_dynamic_range:
        if save_min_max:
            vmin, vmax = x_real.min().item(), x_real.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_real.{idx}'] = [vmin, vmax]
            vmin, vmax = x_im.min().item(), x_im.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_im.{idx}'] = [vmin, vmax]
        else:
            vmax = x_real.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_real.{idx}'] = vmax
            vmax = x_im.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_im.{idx}'] = vmax

    if static_quant:
        if not symmetric:
            x_real = asymmetric_fake_quant(x_real, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.{name}_real.{idx}'],f'model.layers.{layer_idx}.self_attn.{name}_real.{idx}')
            x_im = asymmetric_fake_quant(x_im, dynamic_range_dict_load[f'model.layers.{layer_idx}.self_attn.{name}_im.{idx}'],f'model.layers.{layer_idx}.self_attn.{name}_im.{idx}')
        # TODO: if symmetric:

    # x_prod_real = x_real*rope_real - x_im * rope_im
    # x_prod_im = x_real*rope_im + x_im*rope_real


    x_prod_real = x_real*rope_real - x_im * rope_im
    x_prod_im = x_real*rope_im + x_im*rope_real
    if save_activation_dynamic_range:
        if save_min_max:
            vmin, vmax = (x_real * rope_real).min().item(), (x_real * rope_real).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_real*rope_real.{idx}'] = [vmin, vmax]
            vmin, vmax = (x_im * rope_im).min().item(), (x_im * rope_im).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_im*rope_im.{idx}'] = [vmin, vmax]
            vmin, vmax = (x_real * rope_im).min().item(), (x_real * rope_im).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_real*rope_im.{idx}'] = [vmin, vmax]
            vmin, vmax = (x_im * rope_real).min().item(), (x_im * rope_real).max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_im*rope_real.{idx}'] = [vmin, vmax]
            vmin, vmax = x_prod_real.min().item(), x_prod_real.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_prod_real.{idx}'] = [vmin, vmax]
            vmin, vmax = x_prod_im.min().item(), x_prod_im.max().item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_prod_im.{idx}'] = [vmin, vmax]
        else:
            vmax = (x_real*rope_real).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_real*rope_real.{idx}'] = vmax
            vmax = (x_im * rope_im).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_im*rope_im.{idx}'] = vmax
            vmax = (x_real*rope_im).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_real*rope_im.{idx}'] = vmax
            vmax = (x_im*rope_real).abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_im*rope_real.{idx}'] = vmax
            vmax = x_prod_real.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_prod_real.{idx}'] = vmax
            vmax = x_prod_im.abs().amax(dim=None).item()
            dynamic_range_dict[f'model.layers.{layer_idx}.self_attn.{name}_prod_im.{idx}'] = vmax


    # TODO: HF need to uses different interleaving
    # x = torch.cat((x_prod_real,x_prod_im),dim=3).view(*x.shape)
    x_embed = torch.cat((x_prod_real,x_prod_im),dim=3).view(*x.shape)
    # print(x_embed.to(dtype=orig_dtype).shape)
    return x_embed.to(dtype=orig_dtype)


class MiniCPMMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = layer_idx

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # if self.layer_idx ==0:
                # print(x)
            if save_activation_dynamic_range:
                if save_min_max:
                    vmin, vmax = x.min().item(), x.max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.up_proj.input'] = [vmin, vmax]
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.gate_proj.input'] = [vmin, vmax]
                    vmin, vmax = self.gate_proj(x).min().item(), self.gate_proj(x).max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.gate_proj.output'] = [vmin, vmax]
                    vmin, vmax = self.up_proj(x).min().item(), self.up_proj(x).max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.up_proj.output'] = [vmin, vmax]
                    vmin, vmax = torch.sigmoid(self.gate_proj(x)).min().item(), torch.sigmoid(self.gate_proj(x)).max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.sigmoid'] = [vmin, vmax]
                    vmin, vmax = self.act_fn(self.gate_proj(x)).min().item(), self.act_fn(self.gate_proj(x)).max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.act(gate).output'] = [vmin, vmax]
                    vmin, vmax = (self.act_fn(self.gate_proj(x)) * self.up_proj(x)).min().item(), (self.act_fn(self.gate_proj(x)) * self.up_proj(x)).max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.down_proj.input'] = [vmin, vmax]
                else:
                    vmax = x.abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.up_proj.input'] = vmax
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.gate_proj.input'] = vmax
                    vmax = self.gate_proj(x).abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.gate_proj.output'] = vmax
                    vmax = self.up_proj(x).abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.up_proj.output'] = vmax
                    vmax = torch.sigmoid(self.gate_proj(x)).abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.sigmoid'] = vmax
                    vmax = self.act_fn(self.gate_proj(x)).abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.act(gate).output'] = vmax
                    vmax = (self.act_fn(self.gate_proj(x)) * self.up_proj(x)).abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.down_proj.input'] = vmax
            
            if static_quant:
                x1 = asymmetric_fake_quant(x, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.mlp.up_proj.input'],f'model.layers.{self.layer_idx}.mlp.up_proj.input')
                gate_x = asymmetric_fake_quant(self.gate_proj(x1), dynamic_range_dict_load[f'model.layers.{self.layer_idx}.mlp.gate_proj.output'], f'model.layers.{self.layer_idx}.mlp.gate_proj.output')
                up_x = asymmetric_fake_quant(self.up_proj(x1), dynamic_range_dict_load[f'model.layers.{self.layer_idx}.mlp.up_proj.output'],f'model.layers.{self.layer_idx}.mlp.up_proj.output')
                # sigmoid = asymmetric_fake_quant(torch.sigmoid(gate_x), dynamic_range_dict_load[f'model.layers.{self.layer_idx}.mlp.sigmoid'])
                sigmoid = torch.sigmoid(gate_x)
                act_gate = asymmetric_fake_quant(gate_x*sigmoid, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.mlp.act(gate).output'],f'model.layers.{self.layer_idx}.mlp.act(gate).output')
                down_in = asymmetric_fake_quant(act_gate * up_x, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.mlp.down_proj.input'],f'model.layers.{self.layer_idx}.mlp.down_proj.input')
                down_proj = self.down_proj(down_in)
                down_proj = asymmetric_fake_quant(down_proj, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.mlp.down_proj.output'],f'model.layers.{self.layer_idx}.mlp.down_proj.output')
            else:
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

            if save_activation_dynamic_range:
                if save_min_max:
                    vmin, vmax = down_proj.min().item(), down_proj.max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.down_proj.output'] = [vmin, vmax]
                else:
                    vmax = down_proj.abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp.down_proj.output'] = vmax
            # if self.layer_idx ==0:
                # print(down_proj)
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MiniCPMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MiniCPMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling.get("factor", None)
            if scaling_type == "linear":
                self.rotary_emb = MiniCPMLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MiniCPMDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "longrope":
                self.rotary_emb = MiniCPMLongRoPE(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    short_factor = self.config.rope_scaling["short_factor"],
                    long_factor = self.config.rope_scaling["long_factor"],
                    base=self.rope_theta,
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"]
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # def prepare_conv(self):
    #     self.q_proj_conv = nn.Conv2d(self.hidden_size, self.num_heads * self.head_dim, 1, bias=False)
    #     self.k_proj_conv = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim, 1, bias=False)
    #     self.v_proj_conv = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim, 1, bias=False)
    #     self.o_proj_conv = nn.Conv2d(self.num_heads * self.head_dim, self.hidden_size, 1, bias=False)

    #     self.q_proj_conv.weight.data.copy_(self.q_proj.weight[:, :, None, None])
    #     self.k_proj_conv.weight.data.copy_(self.k_proj.weight[:, :, None, None])
    #     self.v_proj_conv.weight.data.copy_(self.v_proj.weight[:, :, None, None])
    #     self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

    # def prepare_sha(self):
    #     if not hasattr(self, 'forward_mha'):
    #         self.q_proj_sha = nn.ModuleList([nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False) for _ in range(self.num_heads)])
    #         self.k_proj_sha = nn.ModuleList([nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False) for _ in range(self.num_key_value_heads)])
    #         self.v_proj_sha = nn.ModuleList([nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False) for _ in range(self.num_key_value_heads)])
    #         self.o_proj_conv = nn.Conv2d(self.num_heads * self.head_dim, self.hidden_size, 1, bias=False)
    #         for conv in self.q_proj_sha:
    #             conv.half().cuda()
    #         for conv in self.k_proj_sha:
    #             conv.half().cuda()
    #         for conv in self.v_proj_sha:
    #             conv.half().cuda()
    #         self.o_proj_conv.half().cuda()
    #         self.cache_cat = nn.ModuleList([op.Concat(0) for _ in range(2)])  # 2: (key,value)
    #         self.forward_mha = self.forward
    #         self.forward = self.forward_sha

    #     for i in range(self.num_heads):
    #         self.q_proj_sha[i].weight.data.copy_(self.q_proj.weight[i*self.head_dim:(i+1)*self.head_dim, :, None, None])
    #         self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:,:,None, None])
    #     for i in range(self.num_key_value_heads):
    #         self.k_proj_sha[i].weight.data.copy_(self.k_proj.weight[i*self.head_dim:(i+1)*self.head_dim, :, None, None])
    #         self.v_proj_sha[i].weight.data.copy_(self.v_proj.weight[i*self.head_dim:(i+1)*self.head_dim, :, None, None])

    # def forward_sha(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_value: Optional[Tuple[torch.Tensor]] = None,
    #     output_attentions: bool = False,
    #     use_cache: bool = False,
    #     **kwargs, 
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    #     if "padding_mask" in kwargs:
    #         warnings.warn(
    #             "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
    #         )

    #     bsz, q_len, _ = hidden_states.size()

    #     if save_activation_dynamic_range:
    #         if save_min_max:
    #             vmin, vmax = hidden_states.min().item(), hidden_states.max().item()
    #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_proj.input'] = [vmin, vmax]
    #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_proj.input'] = [vmin, vmax]
    #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.v_proj.input'] = [vmin, vmax]
    #         else:
    #             vmax = hidden_states.abs().amax(dim=None).item()
    #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_proj.input'] = vmax
    #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_proj.input'] = vmax
    #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.v_proj.input'] = vmax
        
    #     hidden_states = torch.reshape(hidden_states, (bsz, -1, 1, self.hidden_size))
    #     hidden_states = hidden_states.transpose(1,3)
    #     query_states = [q_proj(hidden_states).permute(0,2,3,1) for q_proj in self.q_proj_sha]
    #     key_states = [k_proj(hidden_states).permute(0,2,3,1) for k_proj in self.k_proj_sha]
    #     value_states = [v_proj(hidden_states).permute(0,2,3,1) for v_proj in self.v_proj_sha]
    #     # print(len(query_states)) # 24
    #     # print(len(key_states)) # 8
    #     # print(len(value_states)) # 8
    #     # print(query_states[0].shape) # [1,1,225,64]
    #     # print(key_states[0].shape) 
    #     # print(value_states[0].shape)

    #     # NOTE: save_dynamic_range for sha:

    #     for idx, q_state in enumerate(query_states):
    #         if save_activation_dynamic_range:
    #             if save_min_max:
    #                 vmin, vmax = q_state.min().item(), q_state.max().item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_proj.output.{idx}'] = [vmin, vmax]
    #             else:
    #                 vmax = q_state.abs().amax(dim=None).item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_proj.output.{idx}'] = vmax

    #     for idx, k_state in enumerate(key_states):
    #         if save_activation_dynamic_range:
    #             if save_min_max:
    #                 vmin, vmax = k_state.min().item(), k_state.max().item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_proj.output.{idx}'] = [vmin, vmax]
    #             else:
    #                 vmax = k_state.abs().amax(dim=None).item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_proj.output.{idx}'] = vmax

    #     for idx, v_state in enumerate(value_states):
    #         if save_activation_dynamic_range:
    #             if save_min_max:
    #                 vmin, vmax = v_state.min().item(), v_state.max().item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.v_proj.output.{idx}'] = [vmin, vmax]
    #             else:
    #                 vmax = v_state.abs().amax(dim=None).item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.v_proj.output.{idx}'] = vmax


    #     kv_seq_len = key_states[0].shape[-2]
    #     if past_key_value is not None:
    #         if self.layer_idx is None:
    #             raise ValueError(
    #                 f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
    #                 "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
    #                 "with a layer index."
    #             )
    #         kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
    #     cos, sin = self.rotary_emb(value_states[0].to(torch.float32), seq_len=kv_seq_len)

    #     if save_activation_dynamic_range:
    #         for idx, q_state in enumerate(query_states):
    #             if save_min_max:
    #                 vmin, vmax = rotate_half(q_state).min().item(), rotate_half(q_state).max().item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.rotate_q_states.{idx}'] = [vmin, vmax]
    #             else:
    #                 vmax = rotate_half(q_state).abs().amax(dim=None).item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.rotate_q_states.{idx}'] = vmax

    #         for idx, k_state in enumerate(key_states):
    #             if save_min_max:
    #                 vmin, vmax = cos.min().item(), cos.max().item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.cos'] = [vmin, vmax]
    #                 vmin, vmax = sin.min().item(), sin.max().item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.sin'] = [vmin, vmax]

    #                 vmin, vmax = rotate_half(k_state).min().item(), rotate_half(k_state).max().item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.rotate_k_states.{idx}'] = [vmin, vmax]
    #             else:
    #                 vmax = cos.abs().amax(dim=None).item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.cos'] = vmax
    #                 vmax = sin.abs().amax(dim=None).item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.sin'] = vmax

    #                 vmax = rotate_half(k_state).abs().amax(dim=None).item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.rotate_k_states.{idx}'] = vmax

    #         if use_qualcomm:
    #             query_states = [apply_rotary_pos_emb_qualcomm_single("q", q, cos, sin, position_ids, self.layer_idx, idx) for idx, q in enumerate(query_states)]
    #             key_states = [apply_rotary_pos_emb_qualcomm_single("k", k, cos, sin, position_ids, self.layer_idx, idx) for idx, k in enumerate(key_states)]
    #         else: # NOT ready
    #             query_states = [apply_rotary_pos_emb(q, q, cos, sin, position_ids, self.layer_idx, idx)[0] for idx, q in enumerate(query_states)]
    #             key_states = [apply_rotary_pos_emb(k, k, cos, sin, position_ids, self.layer_idx, idx)[0] for idx, k in enumerate(key_states)]
    #     else:
    #         if use_qualcomm:
    #             query_states = [apply_rotary_pos_emb_qualcomm_single("q", q, cos, sin, position_ids, idx) for idx, q in enumerate(query_states)]
    #             key_states = [apply_rotary_pos_emb_qualcomm_single("k", k, cos, sin, position_ids, idx) for idx, k in enumerate(key_states)]
    #         else: # NOT ready
    #             query_states = [apply_rotary_pos_emb(q, q, cos, sin, position_ids, idx)[0] for idx, q in enumerate(query_states)]
    #             key_states = [apply_rotary_pos_emb(k, k, cos, sin, position_ids, idx)[0] for idx, k in enumerate(key_states)]
    #     # NOTE: not sure 
    #     # key_states = [k.transpose(2, 3) for k in key_states]

    #     if save_activation_dynamic_range:
    #         if save_min_max:
    #             for idx, q_state in enumerate(query_states):
    #                 vmin, vmax = q_state.min().item(), q_state.max().item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_rotary_emb.output.{idx}'] = [vmin, vmax]
    #             for idx, k_state in enumerate(key_states):
    #                 vmin, vmax = k_state.min().item(), k_state.max().item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_rotary_emb.output.{idx}'] = [vmin, vmax]
    #         else:
    #             for idx, q_state in enumerate(query_states):
    #                 vmax = q_state.abs().amax(dim=None).item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_rotary_emb.output.{idx}'] = vmax
    #             for idx, k_state in enumerate(key_states):
    #                 vmax = k_state.abs().amax(dim=None).item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_rotary_emb.output.{idx}'] = vmax
        
    #     # if self.return_new_key_value_only:
    #     #     present_key_value = (tuple(key_states), tuple(value_states)) if use_cache else None
    #     #     if self.concat_head_in_batch_dimension:
    #     #         present_key_value = tuple(cat(*present) for cat, present in zip(self.cache_cat, present_key_value)) if use_cache else None

    #     # if past_key_value is not None:
    #     #     # print(past_key_value)
    #     #     # reuse k, v, self_attention
    #     #     # if self.concat_head_in_batch_dimension:
    #     #     if False:
    #     #         past_key, past_value = [[past[head:head + 1, ...] for head in range(self.num_key_value_heads)] for past in past_key_value]
    #     #     else:
    #     #         past_key, past_value = past_key_value
    #     #     key_states = [torch.cat([pk, k], dim=3) for pk, k in zip(past_key, key_states)]
    #     #     value_states = [torch.cat([pv, v], dim=2) for pv, v in zip(past_value, value_states)]
    #     #     if self.layer_idx == 0:
    #     #         past_key_value. += key_states.shape[-2]            
    #     # FIXME: 弃用，维护多个麻烦

    #     # if past_key_value is not None:
    #     #     cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
    #     #     new_k_states = []
    #     #     new_v_states = []
    #     #     for idx, (k,v) in enumerate(zip(key_states, value_states)):
    #     #         update_k, update_v = past_key_value.update_fix(k, v, self.layer_idx, idx, cache_kwargs)
    #     #         new_k_states.append(update_k)
    #     #         new_v_states.append(update_v)
    #     #     key_states = new_k_states
    #     #     value_states = new_v_states
    #     #     for k in key_states:
    #     #         print(k.shape)
    #     #     print(len(key_states))

    #     # NOTE: 不需要了。因为use_cache=False
    #     # if save_activation_dynamic_range:
    #     #     if save_min_max:
    #     #         for idx, k_state in enumerate(key_states):
    #     #             vmin, vmax = k_state.min().item(), k_state.max().item()
    #     #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.concat_k_states.{idx}'] = [vmin, vmax]
    #     #         for idx, v_state in enumerate(value_states):
    #     #             vmin, vmax = v_state.min().item(), v_state.max().item()
    #     #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.concat_v_states.{idx}'] = [vmin, vmax]
    #     #     else:
    #     #         for idx, k_state in enumerate(key_states):
    #     #             vmax = k_state.abs().amax(dim=None).item()
    #     #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.concat_k_states.{idx}'] = vmax
    #     #         for idx, v_state in enumerate(value_states):
    #     #             vmax = v_state.abs().amax(dim=None).item()
    #     #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.concat_v_states.{idx}'] = vmax
        
    #     # NOTE: repeat kv
    #     new_key_states = []
    #     new_value_states = []
    #     for key_state in key_states:
    #         new_key_states.extend([key_state]*self.num_key_value_groups)
    #     for value_state in value_states:
    #         new_value_states.extend([value_state]*self.num_key_value_groups)
    #     key_states = new_key_states
    #     value_states = new_value_states
    #     # if not self.return_new_key_value_only:
    #     #     present_key_value = (tuple(key_states), tuple(value_states)) if use_cache else None
    #     attn_weights = [torch.matmul(q, k.transpose(2,3)) / math.sqrt(self.head_dim) for q, k in zip(query_states, key_states)]
    #     if save_activation_dynamic_range:
    #         if save_min_max:
    #             for idx, attn_matrix in enumerate(attn_weights):
    #                 vmin, vmax = attn_matrix.min().item(), attn_matrix.max().item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.attn_weights.{idx}'] = [vmin, vmax]
    #         else:
    #             for idx, attn_matrix in enumerate(attn_weights):
    #                 vmax = attn_matrix.abs().amax(dim=None).item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.attn_weights.{idx}'] = vmax

    #     if attn_weights[0].size() != (bsz, 1, q_len, kv_seq_len):
    #         raise ValueError(
    #             f"Attention weights should be of size {(bsz, 1, q_len, kv_seq_len)}, but is"
    #             f" {attn_weights[0].size()}"
    #         )

    #     if attention_mask is not None:
    #         if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    #             raise ValueError(
    #                 f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
    #             )
    #         if save_activation_dynamic_range:
    #             if save_min_max:
    #                 vmin, vmax = attention_mask.min().item(), attention_mask.max().item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.causal_mask'] = [vmin, vmax]
    #             else:
    #                 vmax = attention_mask.abs().amax(dim=None).item()
    #                 dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.causal_mask'] = vmax

    #         attn_weights = [aw + attention_mask for aw in attn_weights]

    #         if save_activation_dynamic_range:
    #             if save_min_max:
    #                 for idx in range(len(attn_weights)):
    #                     head_attn_weights = attn_weights[idx]
    #                     vmin, vmax = head_attn_weights.min().item(), head_attn_weights.max().item()
    #                     dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.attn_weights_add_mask.{idx}'] = [vmin, vmax]
    #             else:
    #                 for idx in range(len(attn_weights)):
    #                     head_attn_weights = attn_weights[idx]
    #                     vmax = head_attn_weights.abs().amax(dim=None).item()
    #                     dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.attn_weights_add_mask.{idx}'] = vmax

    #     # upcast attention to fp32
    #     attn_weights = [nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(query_states[0].dtype) for aw in attn_weights]
    #     for idx, weight in enumerate(attn_weights):
    #         key = f'model.layers.{self.layer_idx}.self_attn.softmax.output.{idx}'
    #         if save_min_max:
    #             vmin, vmax = weight.min().item(), weight.max().item()
    #             dynamic_range_dict[key] = [vmin, vmax]
    #         else:
    #             vmax = weight.abs().amax(dim=None).item()
    #             dynamic_range_dict[key] = vmax
                
    #     attn_output = [torch.matmul(aw, v) for aw, v in zip(attn_weights, value_states)]
    #     for idx, weight in enumerate(attn_output):
    #         key = f'model.layers.{self.layer_idx}.self_attn.attn_output.{idx}'
    #         if save_min_max:
    #             vmin, vmax = weight.min().item(), weight.max().item()
    #             dynamic_range_dict[key] = [vmin, vmax]
    #         else:
    #             vmax = weight.abs().amax(dim=None).item()
    #             dynamic_range_dict[key] = vmax

    #     if attn_output[0].size() != (bsz, 1, q_len, self.head_dim):
    #         raise ValueError(
    #             f"`attn_output` should be of size {(bsz, 1, q_len, self.head_dim)}, but is"
    #             f" {attn_output[0].size()}"
    #         )

    #     attn_output = torch.cat(attn_output, dim=3)
    #     attn_output = attn_output.permute(0, 3, 1, 2)
    #     if save_activation_dynamic_range:
    #         if save_min_max:
    #             vmin, vmax = attn_output.min().item(), attn_output.max().item()
    #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.o_proj.input'] = [vmin, vmax]
    #         else:
    #             vmax = attn_output.abs().amax(dim=None).item()
    #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.o_proj.input'] = vmax

    #     attn_output = self.o_proj_conv(attn_output)
    #     attn_output = attn_output.transpose(1,3)
    #     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    #     if save_activation_dynamic_range:
    #         if save_min_max:
    #             vmin, vmax = attn_output.min().item(), attn_output.max().item()
    #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.o_proj.output'] = [vmin, vmax]
    #         else:
    #             vmax = attn_output.abs().amax(dim=None).item()
    #             dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.o_proj.output'] = vmax

    #     if not output_attentions:
    #         attn_weights = None

    #     return attn_output, attn_weights, past_key_value

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            # if self.layer_idx ==0:
                # print(hidden_states.shape)
                # print(hidden_states.tolist())
            # print(hidden_states)
            if save_activation_dynamic_range:
                if save_min_max:
                    vmin, vmax = hidden_states.min().item(), hidden_states.max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_proj.input'] = [vmin, vmax]
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_proj.input'] = [vmin, vmax]
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.v_proj.input'] = [vmin, vmax]
                else:
                    vmax = hidden_states.abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_proj.input'] = vmax
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_proj.input'] = vmax
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.v_proj.input'] = vmax

            if static_quant:
                hidden_states = asymmetric_fake_quant(hidden_states, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.q_proj.input'],f'model.layers.{self.layer_idx}.self_attn.q_proj.input')

            query_states = self.q_proj(hidden_states) # 在
            # print(query_states)
            # exit(0)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if static_quant:
                query_states = asymmetric_fake_quant(query_states, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.q_proj.output'],f'model.layers.{self.layer_idx}.self_attn.q_proj.output')
                key_states = asymmetric_fake_quant(key_states, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.k_proj.output'],f'model.layers.{self.layer_idx}.self_attn.k_proj.output')
                value_states = asymmetric_fake_quant(value_states, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.v_proj.output'],f'model.layers.{self.layer_idx}.self_attn.v_proj.output')

            if save_activation_dynamic_range:
                if save_min_max:
                    vmin, vmax = query_states.min().item(), query_states.max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_proj.output'] = [vmin, vmax]
                    vmin, vmax = key_states.min().item(), key_states.max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_proj.output'] = [vmin, vmax]
                    vmin, vmax = value_states.min().item(), value_states.max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.v_proj.output'] = [vmin, vmax]
                else:
                    vmax = query_states.abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_proj.output'] = vmax
                    vmax = key_states.abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_proj.output'] = vmax
                    vmax = value_states.abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.v_proj.output'] = vmax
            # if self.layer_idx ==0:
                # print(query_states, key_states, value_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        # print(value_states.shape) # 1,8,677,64
        # print(query_states.shape) # 1,24,677,64
        # print(key_states.shape)
        # print(cos.shape) # 677, 64
        # print(sin.shape) # 677, 64
        if static_quant:
            cos = asymmetric_fake_quant(cos, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.cos'],f'model.layers.{self.layer_idx}.self_attn.cos')
            sin = asymmetric_fake_quant(sin, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.sin'],f'model.layers.{self.layer_idx}.self_attn.sin')

        if save_activation_dynamic_range:
            if save_min_max:
                vmin, vmax = cos.min().item(), cos.max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.cos'] = [vmin, vmax]
                vmin, vmax = sin.min().item(), sin.max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.sin'] = [vmin, vmax]
                vmin, vmax = rotate_half(query_states).min().item(), rotate_half(query_states).max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.rotate_q_states'] = [vmin, vmax]
                vmin, vmax = rotate_half(key_states).min().item(), rotate_half(key_states).max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.rotate_k_states'] = [vmin, vmax] # not used when use_qualcomm
            else:
                vmax = cos.abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.cos'] = vmax
                vmax = sin.abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.sin'] = vmax
                vmax = rotate_half(query_states).abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.rotate_q_states'] = vmax
                vmax = rotate_half(key_states).abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.rotate_k_states'] = vmax
            if use_qualcomm:
                query_states, key_states = apply_rotary_pos_emb_qualcomm(query_states, key_states, cos, sin, position_ids, self.layer_idx)
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, self.layer_idx)
        else:
            if use_qualcomm:
                query_states, key_states = apply_rotary_pos_emb_qualcomm(query_states, key_states, cos, sin, position_ids, self.layer_idx)
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, self.layer_idx)
            
        if save_activation_dynamic_range:
            if save_min_max:
                vmin, vmax = (query_states).min().item(), (query_states).max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_rotary_emb.output'] = [vmin, vmax]
                vmin, vmax = (key_states).min().item(), (key_states).max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_rotary_emb.output'] = [vmin, vmax]
            else: 
                vmax = (query_states).abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.q_rotary_emb.output'] = vmax
                vmax = (key_states).abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.k_rotary_emb.output'] = vmax

        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if static_quant:
            query_states = asymmetric_fake_quant(query_states, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.q_rotary_emb.output'],f'model.layers.{self.layer_idx}.self_attn.q_rotary_emb.output')
            key_states = asymmetric_fake_quant(key_states, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.k_rotary_emb.output'],f'model.layers.{self.layer_idx}.self_attn.k_rotary_emb.output')
            
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if save_activation_dynamic_range:
            if save_min_max:
                vmin, vmax = (key_states).min().item(), (key_states).max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.concat_k_states'] = [vmin, vmax]
                vmin, vmax = (value_states).min().item(), (value_states).max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.concat_v_states'] = [vmin, vmax]
            else: 
                vmax = key_states.abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.concat_k_states'] = vmax
                vmax = value_states.abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.concat_v_states'] = vmax
        
        if static_quant:
            key_states = asymmetric_fake_quant(key_states, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.concat_k_states'],f'model.layers.{self.layer_idx}.self_attn.concat_k_states')
            value_states = asymmetric_fake_quant(value_states, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.concat_v_states'],f'model.layers.{self.layer_idx}.self_attn.concat_v_states')

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if save_activation_dynamic_range:
            if save_min_max:
                vmin, vmax = attn_weights.min().item(), attn_weights.max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.attn_weights'] = [vmin, vmax]
            else:
                vmax = attn_weights.abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.attn_weights'] = vmax
        
        if static_quant:
            attn_weights = asymmetric_fake_quant(attn_weights, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.attn_weights'],f'model.layers.{self.layer_idx}.self_attn.attn_weights')

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            if save_activation_dynamic_range:
                if save_min_max:
                    vmin, vmax = attention_mask.min().item(), attention_mask.max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.causal_mask'] = [vmin, vmax]
                else:
                    vmax = attention_mask.abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.causal_mask'] = vmax
                    
            # if static_quant:
                # attention_mask = asymmetric_fake_quant(attention_mask, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.causal_mask'],f'model.layers.{self.layer_idx}.self_attn.causal_mask')
            
            
            attn_weights = attn_weights + attention_mask

            if save_activation_dynamic_range:
                if save_min_max:
                    vmin, vmax = attn_weights.min().item(), attn_weights.max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.attn_weights_add_mask'] = [vmin, vmax]
                else:
                    vmax = attn_weights.abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.attn_weights_add_mask'] = vmax

            # if static_quant:
            #     attn_weights = asymmetric_fake_quant(attn_weights, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.attn_weights_add_mask'],f'model.layers.{self.layer_idx}.self_attn.attn_weights_add_mask')

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # attn_output = torch.matmul(attn_weights, value_states)
        if save_activation_dynamic_range:
            if save_min_max:
                vmin, vmax = attn_weights.min().item(), attn_weights.max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.softmax.output'] = [vmin, vmax]
            else:
                vmax = attn_weights.abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.softmax.output'] = vmax
        
        if static_quant:
            attn_weights = asymmetric_fake_quant(attn_weights, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.softmax.output'],f'model.layers.{self.layer_idx}.self_attn.softmax.output')

        attn_output = torch.matmul(attn_weights, value_states)

        if save_activation_dynamic_range:
            if save_min_max:
                vmin, vmax = attn_output.min().item(), attn_output.max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.attn_output'] = [vmin, vmax]
            else:
                vmax = attn_output.abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.attn_output'] = vmax

        if static_quant:
            attn_output = asymmetric_fake_quant(attn_output, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.attn_output'],f'model.layers.{self.layer_idx}.self_attn.attn_output')

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            # attn_output = self.o_proj(attn_output)
            if save_activation_dynamic_range:
                if save_min_max:
                    vmin, vmax = attn_output.min().item(), attn_output.max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.o_proj.input'] = [vmin, vmax]
                else:
                    vmax = attn_output.abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.o_proj.input'] = vmax
            attn_output = self.o_proj(attn_output)
            if save_activation_dynamic_range:
                if save_min_max:
                    vmin, vmax = attn_output.min().item(), attn_output.max().item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.o_proj.output'] = [vmin, vmax]
                else:
                    vmax = attn_output.abs().amax(dim=None).item()
                    dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn.o_proj.output'] = vmax

            if static_quant:
                attn_output = asymmetric_fake_quant(attn_output, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn.o_proj.output'],f'model.layers.{self.layer_idx}.self_attn.o_proj.output')

        if not output_attentions:
            attn_weights = None
        # if self.layer_idx ==1:
        #     print(attn_output)
        return attn_output, attn_weights, past_key_value


class MiniCPMFlashAttention2(MiniCPMAttention):
    """
    MiniCPM flash attention module. This module inherits from `MiniCPMAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # MiniCPMFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (MiniCPMRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in MiniCPMFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MiniCPMSdpaAttention(MiniCPMAttention):
    """
    MiniCPM attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MiniCPMAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MiniCPMAttention.forward
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MiniCPMModel is using MiniCPMSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        # print(type(self.q_proj))
        # print(query_states)
        
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


MINICPM_ATTENTION_CLASSES = {
    "eager": MiniCPMAttention,
    "flash_attention_2": MiniCPMFlashAttention2,
    "sdpa": MiniCPMSdpaAttention,
}


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, config: MiniCPMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        config._attn_implementation = "eager"
        self.self_attn = MINICPM_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = MiniCPMMLP(config, layer_idx)
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        # if self.layer_idx == 1:
        #     print("1: ", hidden_states)
        #     # 检查 hidden_states 中是否存在 nan
        #     if torch.isnan(hidden_states).any():
        #         print(f"Exiting at layer {self.layer_idx} due to NaN values.")
        #         exit(0)
        residual = hidden_states
        # if self.layer_idx == 1:
            # print(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        # if self.layer_idx == 1:
            # print(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        if save_activation_dynamic_range:
            if save_min_max:
                vmin, vmax = hidden_states.min().item(), hidden_states.max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn_add_residual'] = [vmin, vmax]
            else:
                vmax = hidden_states.abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.self_attn_add_residual'] = vmax

        if static_quant:
            hidden_states = asymmetric_fake_quant(hidden_states, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.self_attn_add_residual'],f'model.layers.{self.layer_idx}.self_attn_add_residual')

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        if save_activation_dynamic_range:
            if save_min_max:
                vmin, vmax = hidden_states.min().item(), hidden_states.max().item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp_add_residual'] = [vmin, vmax]
            else:
                vmax = hidden_states.abs().amax(dim=None).item()
                dynamic_range_dict[f'model.layers.{self.layer_idx}.mlp_add_residual'] = vmax

        if static_quant:
            hidden_states = asymmetric_fake_quant(hidden_states, dynamic_range_dict_load[f'model.layers.{self.layer_idx}.mlp_add_residual'],f'model.layers.{self.layer_idx}.mlp_add_residual')

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


MINICPM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MiniCPMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPMPreTrainedModel(PreTrainedModel):
    config_class = MiniCPMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniCPMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MINICPM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPMModel(MiniCPMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiniCPMDecoderLayer`]

    Args:
        config: MiniCPMConfig
    """

    def __init__(self, config: MiniCPMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniCPMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # print(input_ids)
        # print(inputs_embeds)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds
        vmin, vmax = hidden_states.min().item(), hidden_states.max().item()
        # print(vmin, vmax)
        if save_activation_dynamic_range:
            if save_min_max:
                vmin, vmax = hidden_states.min().item(), hidden_states.max().item()
                dynamic_range_dict['model.inputs_embeds'] = [vmin, vmax]
            else:
                vmax = hidden_states.abs().amax(dim=None).item()
                dynamic_range_dict['model.inputs_embeds'] = vmax

        if static_quant:
            hidden_states = asymmetric_fake_quant(hidden_states, dynamic_range_dict_load[f'model.inputs_embeds'],f'model.inputs_embeds')
        # print(hidden_states)
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # print(hidden_states)
            # exit(0)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            # print(hidden_states)
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        # print(hidden_states)
        hidden_states = self.norm(hidden_states)
        # print(hidden_states)
        # print("1: ", hidden_states)
        # # 检查 hidden_states 中是否存在 nan
        # if torch.isnan(hidden_states).any():
        #     exit(0)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MiniCPMForCausalLM(MiniCPMPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniCPMForCausalLM

        >>> model = MiniCPMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids) # [0]
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if save_activation_dynamic_range:
                if save_min_max:
                    vmin, vmax = (hidden_states / (self.config.hidden_size / self.config.dim_model_base)).min().item(), (hidden_states / (self.config.hidden_size / self.config.dim_model_base)).max().item()
                    dynamic_range_dict[f'lm_head.input'] = [vmin, vmax]
                else:
                    vmax = (hidden_states / (self.config.hidden_size / self.config.dim_model_base)).abs().amax(dim=None).item()
                    dynamic_range_dict[f'lm_head.input'] = vmax

            if static_quant:
                hidden_states = asymmetric_fake_quant(hidden_states / (self.config.hidden_size / self.config.dim_model_base), dynamic_range_dict_load[f'lm_head.input'],f'lm_head.input')

            # logits = self.lm_head(hidden_states / (self.config.hidden_size / self.config.dim_model_base))
            logits = self.lm_head(hidden_states)

            if save_activation_dynamic_range:
                if save_min_max:
                    vmin, vmax = logits.min().item(), logits.max().item()
                    dynamic_range_dict[f'lm_head.output'] = [vmin, vmax]
                else:
                    vmax = logits.abs().amax(dim=None).item()
                    dynamic_range_dict[f'lm_head.output'] = vmax
            if static_quant:
                logits = asymmetric_fake_quant(logits, dynamic_range_dict_load[f'lm_head.output'],f'lm_head.output')

        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if save_activation_dynamic_range:
            import os
            import json
            from datetime import datetime
            import uuid

            # 获取当前时间并格式化为字符串
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 生成唯一的标识符
            unique_id = str(uuid.uuid4())

            # 构建文件名
            filename = f"llm_activation_dynamic_range_{current_time}_{unique_id}.json"
            # filename = f"activation_dynamic_range_{current_time}.json"
            with open(os.path.join("/home/workspace/code/git/AutoGPTQ_mlm/auto_gptq/activate/", filename), 'w') as f:
                json.dump(dynamic_range_dict, f)
                

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):  
        # past_key_values=None # NOTE: set to None在使用vit的embedding的时候会导致维度不匹配
        # print(past_key_values)
        if past_key_values is not None:
            # print(past_key_values)
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    @torch.inference_mode()
    def chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 4096, num_beams=1, do_sample=True, top_p=0.8, temperature=0.3, logits_processor=None,
             **kwargs):
        if history is None:
            history = []
        if logits_processor:
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                          "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        else:
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                          "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(history_str, return_tensors='pt').to(self.device)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = tokenizer.decode(outputs)
        pattern = re.compile(r".*?(?=<AI>|<用户>)", re.DOTALL)
        matches = pattern.findall(response)
        if len(matches) > 0:
            response = matches[0]
        history.append({"role": "assistant", "content": response})
        return response, history


@add_start_docstrings(
    """
    The MiniCPM Model transformer with a sequence classification head on top (linear layer).

    [`MiniCPMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MINICPM_START_DOCSTRING,
)
class MiniCPMForSequenceClassification(MiniCPMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MiniCPMModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )